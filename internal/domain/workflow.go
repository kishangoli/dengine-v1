package domain

import "time"

type Workflow struct {
    ID        string    `json:"id"`
    Name      string    `json:"name"`
    Status    Status    `json:"status"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}