package service

import (
    "context"
    "log"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/repository"

)

type Scheduler struct {
    taskRepo       repository.TaskRepository
    dependencyRepo repository.DependencyRepository
}

func NewScheduler(taskRepo repository.TaskRepository, dependencyRepo repository.DependencyRepository) *Scheduler {
    return &Scheduler{
        taskRepo:       taskRepo,
        dependencyRepo: dependencyRepo,
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

func (s *Scheduler) run(ctx context.Context) error {
    log.Println("Scheduler running")

    tasks, err := s.taskRepo.GetPendingTasks(ctx)
    if err != nil {
        log.Printf("Error querying pending tasks: %v", err)
        return err
    }
    log.Printf("Scheduler found %d pending tasks", len(tasks))

    for _, task := range tasks {
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
        }
    }

    return nil
}