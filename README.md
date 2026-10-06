# dengine-v1

`dengine-v1` is a Go engine for running LLM workflows made of dependent tasks. Submit a workflow as JSON; the engine stores it in SQLite, runs each task when its dependencies finish, and retries failed work. The current task types—extract, summarize, and classify—call an external LLM API. A browser UI shows the task graph and live status updates.

## Quick start (local)

Set `OPENAI_API_KEY`, then run from the repository root:

```bash
go run ./cmd/dengine/
```

This starts the HTTP server and background workers. Open http://localhost:8080/ui/ to submit and inspect workflows.

## Coming soon

A local C++ inference engine will let workflows run a model on the same machine. The standalone JSON request/response process is already in place; model inference and Go integration are the next steps. See [the native component README](native/README.md).
