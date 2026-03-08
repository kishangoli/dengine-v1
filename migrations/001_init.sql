CREATE TABLE IF NOT EXISTS workflows (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    workflow_id  TEXT NOT NULL,
    type         TEXT NOT NULL,
    input        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    output       TEXT,
    retry_count  INTEGER NOT NULL DEFAULT 0,
    max_retries  INTEGER NOT NULL DEFAULT 3,
    timeout_ns   INTEGER NOT NULL DEFAULT 30000000000,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
);

-- Dependencies table: each row is a DAG edge
CREATE TABLE IF NOT EXISTS task_dependencies (
    task_id       TEXT NOT NULL,
    depends_on_id TEXT NOT NULL,
    PRIMARY KEY (task_id, depends_on_id),
    FOREIGN KEY (task_id) REFERENCES tasks(id),
    FOREIGN KEY (depends_on_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS leases (
    task_id    TEXT PRIMARY KEY,
    worker_id  TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id  TEXT NOT NULL,
    task_id    TEXT NOT NULL,
    timestamp  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (worker_id, task_id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);