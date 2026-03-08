package repository

import (
    "context"
    "github.com/kishangoli/dengine-v1/internal/domain"
)

type WorkflowRepository interface {
    CreateWorkflow(ctx context.Context, workflow *domain.Workflow) error
    GetWorkflow(ctx context.Context, id string) (*domain.Workflow, error)
    UpdateWorkflowStatus(ctx context.Context, id string, status domain.Status) error
}

type TaskRepository interface {
    CreateTask(ctx context.Context, task *domain.Task) error
    GetTask(ctx context.Context, id string) (*domain.Task, error)
    GetTasksByWorkflow(ctx context.Context, workflowID string) ([]*domain.Task, error)
    GetRunnableTasks(ctx context.Context) ([]*domain.Task, error)
    UpdateTaskStatus(ctx context.Context, id string, status domain.Status) error
    UpdateTaskOutput(ctx context.Context, id string, output string) error
    IncrementRetryCount(ctx context.Context, id string) error
}

type DependencyRepository interface {
    CreateDependency(ctx context.Context, dep *domain.TaskDependency) error
    GetDependencies(ctx context.Context, taskID string) ([]*domain.TaskDependency, error)
    AreDependenciesMet(ctx context.Context, taskID string) (bool, error)
}

type LeaseRepository interface {
    AcquireLease(ctx context.Context, lease *domain.Lease) (bool, error)
    RenewLease(ctx context.Context, taskID string, workerID string, newExpiry interface{}) error
    ReleaseLease(ctx context.Context, taskID string) error
    GetExpiredLeases(ctx context.Context) ([]*domain.Lease, error)
    RecordHeartbeat(ctx context.Context, heartbeat *domain.WorkerHeartbeat) error
}