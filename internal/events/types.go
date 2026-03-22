package events

import "time"

type Type string

const (
    TypeTaskUpdated Type = "task.updated"
)

type Envelope struct {
    Type       Type      `json:"type"`
    WorkflowID string    `json:"workflow_id"`
    TaskID     string    `json:"task_id,omitempty"`
    At         time.Time `json:"at"`
    Payload    any       `json:"payload"`
}

type TaskUpdated struct {
    TaskID      string     `json:"task_id"`
    Status      string     `json:"status"`
    Attempts    int        `json:"attempts"`
    MaxAttempts int        `json:"max_attempts"`
    LastError   *string    `json:"last_error,omitempty"`
    NextRunAt   *time.Time `json:"next_run_at,omitempty"`
    Output      *string    `json:"output,omitempty"`
}