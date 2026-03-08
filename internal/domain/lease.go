package domain

import "time"

type Lease struct {
    TaskID    string    `json:"task_id"`
    WorkerID  string    `json:"worker_id"`
    ExpiresAt time.Time `json:"expires_at"`
    CreatedAt time.Time `json:"created_at"`
}