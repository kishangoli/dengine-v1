package service

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/repository"
    "github.com/kishangoli/dengine-v1/internal/executor"

)

type Worker struct {
    workerID string

    taskRepo  repository.TaskRepository
    leaseRepo repository.LeaseRepository
    executors *executor.Registry
}

func NewWorker(workerID string, taskRepo repository.TaskRepository, leaseRepo repository.LeaseRepository, executors *executor.Registry) *Worker {
    return &Worker{
        workerID:  workerID,
        taskRepo:  taskRepo,
        leaseRepo: leaseRepo,
        executors:  executors,
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

    if err := w.executeTask(ctx, task); err != nil {
        _ = w.taskRepo.SetTaskError(ctx, task.ID, err.Error())
        _ = w.taskRepo.IncrementTaskAttempts(ctx, task.ID)

        attempts := task.Attempts + 1
        maxAttempts := task.MaxAttempts
        if maxAttempts == 0 {
            maxAttempts = 3
        }

        if attempts >= maxAttempts {
            _ = w.taskRepo.UpdateTaskStatus(ctx, task.ID, domain.StatusFailed)
            return fmt.Errorf("task execution failed (max attempts reached): %w", err)
        }

        delay := time.Duration(1<<max(0, attempts-1)) * 2 * time.Second
        if delay > 60*time.Second {
            delay = 60 * time.Second
        }
        next := time.Now().Add(delay)

        _ = w.taskRepo.SetTaskNextRunAt(ctx, task.ID, &next)
        _ = w.taskRepo.MarkTaskRunnable(ctx, task.ID)
        return fmt.Errorf("task execution failed (will retry): %w", err)
    }

    _ = w.taskRepo.ClearTaskError(ctx, task.ID)
    _ = w.taskRepo.SetTaskNextRunAt(ctx, task.ID, nil)

    if err := w.taskRepo.UpdateTaskStatus(ctx, task.ID, domain.StatusCompleted); err != nil {
        return fmt.Errorf("failed to update task status to completed: %w", err)
    }

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

    res, err := w.executors.Execute(ctx, task)
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

    return nil
}