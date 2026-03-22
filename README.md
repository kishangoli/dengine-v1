# dengine-v1

A lightweight workflow/task execution engine with a simple web UI to submit LLM workflows and visualize LLM task graphs with dependency-aware scheduling. Designed for local demos.

## Requirements

- Go (use the version specified in `go.mod`)
- SQLite (binary optional; the app uses SQLite via driver)
- An OpenAI API key (for LLM-backed task types)

## Quick start (local)

From the repo root:

```bash
go run ./cmd/dengine/
```

This boots the HTTP server and starts the background workers.

### Open the UI

- http://localhost:8080/ui/

## Environment variables

At minimum, set:

- `OPENAI_API_KEY` – required for LLM task executors (e.g. `extract`, `summarize`, `classify`)

Optional (defaults depend on the code):

- `API_PORT` – server port (commonly `8080`)
- `DB_PATH` – path to the SQLite DB file (defaults are typically fine for local)

Example:

```bash
export OPENAI_API_KEY="your_key_here"
export API_PORT=8080
export DB_PATH="./dengine.db"

go run ./cmd/dengine/
```

## Submitting a workflow

Use the UI to paste JSON and submit, or call the API directly (endpoints depend on your router setup). The UI also includes a graph view to visualize task dependencies and statuses.

## Task execution & dependencies (important behavior)

- `depends_on` controls scheduling (a task runs only when its dependencies have completed).
- Downstream tasks receive upstream outputs by having dependency outputs prepended into the task input at execution time.
- If a task type has no executor registered, it will fail.

Note: For demo mode, the scheduler/workers continue running even if a workflow has failed; the server remains active.

## Docker (optional)

A `Dockerfile` is included for convenience. You do **not** need Docker to run locally.

### Build

```bash
docker build -t dengine-demo .
```

### Run

```bash
docker run --rm -p 8080:8080 --env-file .env dengine-demo
```

Then open:

- http://localhost:8080/ui/

### SQLite persistence in Docker

By default the DB is stored inside the container (ephemeral). To persist it on your machine:

```bash
mkdir -p ./data
docker run --rm -p 8080:8080 --env-file .env \
  -e DB_PATH=/data/dengine.db \
  -v "$(pwd)/data:/data" \
  dengine-demo
```
