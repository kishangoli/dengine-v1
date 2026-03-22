package domain

import "time"

type Task struct {
    ID         string        `json:"id"`
    WorkflowID string        `json:"workflow_id"`
    Type       string        `json:"type"`
    Input      string        `json:"input,omitempty"`
    Status     Status        `json:"status"`
    Output     string        `json:"output,omitempty"`
    RetryCount int           `json:"retry_count"`
    MaxRetries int           `json:"max_retries"`
    Timeout    time.Duration `json:"timeout"`
    DependsOn  []string      `json:"depends_on,omitempty"`
    CreatedAt  time.Time     `json:"created_at"`
    UpdatedAt  time.Time     `json:"updated_at"`
    Attempts    int        `json:"attempts"`
    MaxAttempts int        `json:"max_attempts"`
    LastError   *string    `json:"last_error,omitempty"`
    NextRunAt   *time.Time `json:"next_run_at,omitempty"`

}