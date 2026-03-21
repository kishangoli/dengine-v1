package executor

import (
    "context"
    "fmt"

    "github.com/kishangoli/dengine-v1/internal/domain"
)

type Executor interface {
    Execute(ctx context.Context, task *domain.Task) (*Result, error)
}

type Registry struct {
    byType map[string]Executor
}

func NewRegistry() *Registry {
    return &Registry{byType: map[string]Executor{}}
}

func (r *Registry) Register(taskType string, ex Executor) {
    r.byType[taskType] = ex
}

func (r *Registry) Get(taskType string) (Executor, bool) {
    ex, ok := r.byType[taskType]
    return ex, ok
}

func (r *Registry) Execute(ctx context.Context, task *domain.Task) (*Result, error) {
    ex, ok := r.Get(task.Type)
    if !ok {
        return nil, fmt.Errorf("no executor registered for task type %q", task.Type)
    }
    return ex.Execute(ctx, task)
}