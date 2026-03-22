ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tasks ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 3;
ALTER TABLE tasks ADD COLUMN last_error TEXT;

ALTER TABLE tasks ADD COLUMN next_run_at DATETIME;

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_next_run_at ON tasks(next_run_at);

CREATE INDEX IF NOT EXISTS idx_leases_task_id ON leases(task_id);
CREATE INDEX IF NOT EXISTS idx_leases_expires_at ON leases(expires_at);