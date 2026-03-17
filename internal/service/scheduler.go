package service

import (
    "context"
    "log"
    "time"

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