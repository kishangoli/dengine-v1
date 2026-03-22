package service

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "strings"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/events"
    "github.com/kishangoli/dengine-v1/internal/executor"
    "github.com/kishangoli/dengine-v1/internal/repository"
)

type Worker struct {
    workerID string

    taskRepo       repository.TaskRepository
    leaseRepo      repository.LeaseRepository
    dependencyRepo repository.DependencyRepository
    executors      *executor.Registry
    bus            *events.Bus
}

func NewWorker(
    workerID string,
    taskRepo repository.TaskRepository,
    leaseRepo repository.LeaseRepository,
    dependencyRepo repository.DependencyRepository,
    executors *executor.Registry,
    bus *events.Bus,
) *Worker {
    return &Worker{
        workerID:       workerID,
        taskRepo:       taskRepo,
        leaseRepo:      leaseRepo,
        dependencyRepo: dependencyRepo,
        executors:      executors,
        bus:            bus,
    }
}

func (w *Worker) Start(ctx context.Context) {
    log.Printf("Worker %s started", w.workerID)

    ticker := time.NewTicker(500 * time.Millisecond)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            log.Printf("Worker %s stopped", w.workerID)
            return
        case <-ticker.C:
            if err := w.processTask(ctx); err != nil {
                log.Printf("Worker %s error: %v", w.workerID, err)
            }
        }
    }
}

func (w *Worker) processTask(ctx context.Context) (err error) {
    tasks, err := w.taskRepo.GetRunnableTasks(ctx)
    if err != nil {
        return fmt.Errorf("failed to query runnable tasks: %w", err)
    }
    if len(tasks) == 0 {
        return nil
    }

    task := tasks[0]

    lease := &domain.Lease{
        TaskID:    task.ID,
        WorkerID:  w.workerID,
        ExpiresAt: time.Now().Add(1 * time.Minute),
        CreatedAt: time.Now(),
    }

    acquired, err := w.leaseRepo.AcquireLease(ctx, lease)
    if err != nil {
        return fmt.Errorf("failed to acquire lease: %w", err)
    }
    if !acquired {
        return nil
    }

    defer func() {
        _ = w.leaseRepo.ReleaseLease(ctx, task.ID)
    }()

    if err := w.taskRepo.UpdateTaskStatus(ctx, task.ID, domain.StatusRunning); err != nil {
        return fmt.Errorf("failed to update task status to running: %w", err)
    }
    w.publishTask(ctx, task.WorkflowID, task.ID)

    if err := w.executeTask(ctx, task); err != nil {
        _ = w.taskRepo.SetTaskError(ctx, task.ID, err.Error())
        _ = w.taskRepo.IncrementTaskAttempts(ctx, task.ID)

        fresh, getErr := w.taskRepo.GetTask(ctx, task.ID)
        if getErr == nil {
            task = fresh
        }

        attempts := task.Attempts
        maxAttempts := task.MaxAttempts
        if maxAttempts == 0 {
            maxAttempts = 3
        }

        if attempts >= maxAttempts {
            _ = w.taskRepo.UpdateTaskStatus(ctx, task.ID, domain.StatusFailed)
            w.publishTask(ctx, task.WorkflowID, task.ID)
            return fmt.Errorf("task execution failed (max attempts reached): %w", err)
        }

        delay := time.Duration(1<<max(0, attempts-1)) * 2 * time.Second
        if delay > 60*time.Second {
            delay = 60 * time.Second
        }
        next := time.Now().Add(delay)

        _ = w.taskRepo.SetTaskNextRunAt(ctx, task.ID, &next)
        _ = w.taskRepo.MarkTaskRunnable(ctx, task.ID)
        w.publishTask(ctx, task.WorkflowID, task.ID)

        return fmt.Errorf("task execution failed (will retry): %w", err)
    }

    _ = w.taskRepo.ClearTaskError(ctx, task.ID)
    _ = w.taskRepo.SetTaskNextRunAt(ctx, task.ID, nil)

    if err := w.taskRepo.UpdateTaskStatus(ctx, task.ID, domain.StatusCompleted); err != nil {
        return fmt.Errorf("failed to update task status to completed: %w", err)
    }
    w.publishTask(ctx, task.WorkflowID, task.ID)

    return nil
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func (w *Worker) executeTask(ctx context.Context, task *domain.Task) error {
    log.Printf("Worker %s executing task %s of type %s", w.workerID, task.ID, task.Type)

    enriched, err := w.enrichInputWithDependencyOutputs(ctx, task)
    if err != nil {
        return err
    }
    taskToRun := *task
    taskToRun.Input = enriched

    res, err := w.executors.Execute(ctx, &taskToRun)
    if err != nil {
        return err
    }

    b, err := json.Marshal(res)
    if err != nil {
        return fmt.Errorf("failed to marshal executor result: %w", err)
    }

    if err := w.taskRepo.UpdateTaskOutput(ctx, task.ID, string(b)); err != nil {
        return fmt.Errorf("failed to store task output: %w", err)
    }

    w.publishTask(ctx, task.WorkflowID, task.ID)
    return nil
}

func (w *Worker) enrichInputWithDependencyOutputs(ctx context.Context, task *domain.Task) (string, error) {
    if w.dependencyRepo == nil {
        return task.Input, nil
    }

    deps, err := w.dependencyRepo.GetDependencies(ctx, task.ID)
    if err != nil {
        return "", fmt.Errorf("failed to load dependencies for task %s: %w", task.ID, err)
    }
    if len(deps) == 0 {
        return task.Input, nil
    }

    var sb strings.Builder
    sb.WriteString("UPSTREAM TASK OUTPUTS (JSON):\n")

    for _, d := range deps {
        up, err := w.taskRepo.GetTask(ctx, d.DependsOnID)
        if err != nil {
            return "", fmt.Errorf("failed to load upstream task %s for task %s: %w", d.DependsOnID, task.ID, err)
        }

        out := strings.TrimSpace(up.Output)
        if out == "" {
            out = `""`
        }

        sb.WriteString("- ")
        sb.WriteString(up.ID)
        sb.WriteString(" (")
        sb.WriteString(up.Type)
        sb.WriteString("): ")
        sb.WriteString(out)
        sb.WriteString("\n")
    }

    sb.WriteString("\nORIGINAL INPUT:\n")
    sb.WriteString(strings.TrimSpace(task.Input))
    sb.WriteString("\n")

    return sb.String(), nil
}

func (w *Worker) publishTask(ctx context.Context, workflowID string, taskID string) {
    if w.bus == nil {
        return
    }
    t, err := w.taskRepo.GetTask(ctx, taskID)
    if err != nil {
        return
    }
    output := t.Output
    w.bus.Publish(events.Envelope{
        Type:       events.TypeTaskUpdated,
        WorkflowID: workflowID,
        TaskID:     taskID,
        At:         time.Now(),
        Payload: events.TaskUpdated{
            TaskID:      t.ID,
            Status:      string(t.Status),
            Attempts:    t.Attempts,
            MaxAttempts: t.MaxAttempts,
            LastError:   t.LastError,
            NextRunAt:   t.NextRunAt,
            Output:      &output,
        },
    })
}