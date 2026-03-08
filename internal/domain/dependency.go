package domain

type TaskDependency struct {
    TaskID      string `json:"task_id"`
    DependsOnID string `json:"depends_on_id"`
}