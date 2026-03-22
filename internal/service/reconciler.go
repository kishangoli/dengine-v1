package service

import (
    "context"
    "log"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/repository"
)

type Reconciler struct {
    taskRepo  repository.TaskRepository
    leaseRepo repository.LeaseRepository

    interval time.Duration
}

func NewReconciler(taskRepo repository.TaskRepository, leaseRepo repository.LeaseRepository, interval time.Duration) *Reconciler {
    if interval <= 0 {
        interval = 5 * time.Second
    }
    return &Reconciler{
        taskRepo:  taskRepo,
        leaseRepo: leaseRepo,
        interval: interval,
    }
}

func (r *Reconciler) Start(ctx context.Context) {
    log.Printf("Reconciler started")
    ticker := time.NewTicker(r.interval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            log.Printf("Reconciler stopped")
            return
        case <-ticker.C:
            r.reconcileOnce(ctx)
        }
    }
}

func (r *Reconciler) reconcileOnce(ctx context.Context) {
    now := time.Now()

    stale, err := r.taskRepo.GetStaleRunningTasks(ctx, now)
    if err != nil {
        log.Printf("Reconciler error: get stale running tasks: %v", err)
        return
    }

    for _, t := range stale {
        _ = r.leaseRepo.ReleaseLease(ctx, t.ID)

        maxAttempts := t.MaxAttempts
        if maxAttempts == 0 {
            maxAttempts = 3
        }

        if t.Attempts >= maxAttempts {
            _ = r.taskRepo.SetTaskError(ctx, t.ID, "stale running task; max attempts reached")
            _ = r.taskRepo.UpdateTaskStatus(ctx, t.ID, domain.StatusFailed)
            continue
        }

        _ = r.taskRepo.SetTaskNextRunAt(ctx, t.ID, nil)
        _ = r.taskRepo.MarkTaskRunnable(ctx, t.ID)
    }
}