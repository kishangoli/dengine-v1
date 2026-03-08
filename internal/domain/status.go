package domain

type Status string

const (
	StatusPending   Status = "pending"
    StatusRunnable  Status = "runnable"
    StatusRunning   Status = "running"
    StatusCompleted Status = "completed"
    StatusFailed    Status = "failed"
)