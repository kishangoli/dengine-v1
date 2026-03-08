package domain

import "time"

type WorkerHeartbeat struct {
    WorkerID  string    `json:"worker_id"`
    TaskID    string    `json:"task_id"`
    Timestamp time.Time `json:"timestamp"`
}