package executor

import "time"

type Result struct {
    TaskID      string    `json:"task_id"`
    TaskType    string    `json:"task_type"`
    Model       string    `json:"model,omitempty"`
    Text        string    `json:"text"`
    GeneratedAt time.Time `json:"generated_at"`
}