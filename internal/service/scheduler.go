package service

import (
    "context"
    "log"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/events"
    "github.com/kishangoli/dengine-v1/internal/repository"
)

type Scheduler struct {
    taskRepo       repository.TaskRepository
    dependencyRepo repository.DependencyRepository
    bus            *events.Bus
}

func NewScheduler(taskRepo repository.TaskRepository, dependencyRepo repository.DependencyRepository, bus *events.Bus) *Scheduler {
    return &Scheduler{
        taskRepo:       taskRepo,
        dependencyRepo: dependencyRepo,
        bus:            bus,
    }
}

func (s *Scheduler) Start(ctx context.Context) {
    log.Println("Scheduler started")
    ticker := time.NewTicker(1 * time.Second) // Run every 1 second
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            log.Println("Scheduler stopped")
            return
        case <-ticker.C:
            if err := s.run(ctx); err != nil {
                log.Printf("Scheduler error: %v", err)
            }
        }
    }
}

func (s *Scheduler) publishTask(ctx context.Context, workflowID string, taskID string) {
    if s.bus == nil {
        return
    }
    t, err := s.taskRepo.GetTask(ctx, taskID)
    if err != nil {
        return
    }
    s.bus.Publish(events.Envelope{
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
            Output:      &t.Output,
        },
    })
}

func (s *Scheduler) run(ctx context.Context) error {
    log.Println("Scheduler running")

    tasks, err := s.taskRepo.GetPendingTasks(ctx)
    if err != nil {
        log.Printf("Error querying pending tasks: %v", err)
        return err
    }
    log.Printf("Scheduler found %d pending tasks", len(tasks))

    now := time.Now()

    for _, task := range tasks {
        if task.NextRunAt != nil && now.Before(*task.NextRunAt) {
            continue
        }

        depsMet, err := s.dependencyRepo.AreDependenciesMet(ctx, task.ID)
        if err != nil {
            log.Printf("Error checking dependencies for task %s: %v", task.ID, err)
            return err
        }

        if depsMet {
            if err := s.taskRepo.UpdateTaskStatus(ctx, task.ID, domain.StatusRunnable); err != nil {
                log.Printf("Error updating task %s to runnable: %v", task.ID, err)
                return err
            }
            log.Printf("Task %s marked as runnable", task.ID)

            s.publishTask(ctx, task.WorkflowID, task.ID)
        }
    }

    return nil
}