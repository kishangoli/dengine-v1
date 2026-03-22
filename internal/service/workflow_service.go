package service

import (
    "context"
    "errors"
    "fmt"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/repository"
)

type WorkflowService struct {
    workflowRepo   repository.WorkflowRepository
    taskRepo       repository.TaskRepository
    dependencyRepo repository.DependencyRepository
}

func NewWorkflowService(
    workflowRepo repository.WorkflowRepository,
    taskRepo repository.TaskRepository,
    dependencyRepo repository.DependencyRepository,
) *WorkflowService {
    return &WorkflowService{
        workflowRepo:   workflowRepo,
        taskRepo:       taskRepo,
        dependencyRepo: dependencyRepo,
    }
}

func (s *WorkflowService) SubmitWorkflow(ctx context.Context, workflow *domain.Workflow, tasks []*domain.Task) error {
    if err := validateWorkflow(workflow, tasks); err != nil {
        return fmt.Errorf("workflow validation failed: %w", err)
    }

    now := time.Now()

    if workflow.ID == "" {
        return fmt.Errorf("workflow id is required")
    }
    if workflow.Name == "" {
        workflow.Name = workflow.ID
    }
    if workflow.Status == "" {
        workflow.Status = domain.StatusPending
    }
    if workflow.CreatedAt.IsZero() {
        workflow.CreatedAt = now
    }
    workflow.UpdatedAt = now

    if err := s.workflowRepo.CreateWorkflow(ctx, workflow); err != nil {
        return fmt.Errorf("failed to store workflow: %w", err)
    }

    for _, task := range tasks {
        task.WorkflowID = workflow.ID
        task.Status = domain.StatusPending
        task.CreatedAt = now
        task.UpdatedAt = now

        if err := s.taskRepo.CreateTask(ctx, task); err != nil {
            return fmt.Errorf("failed to store task %s: %w", task.ID, err)
        }

        for _, depID := range task.DependsOn {
            dep := &domain.TaskDependency{
                TaskID:      task.ID,
                DependsOnID: depID,
            }
            if err := s.dependencyRepo.CreateDependency(ctx, dep); err != nil {
                return fmt.Errorf("failed to store dependency for task %s: %w", task.ID, err)
            }
        }
    }

    return nil
}

func validateWorkflow(workflow *domain.Workflow, tasks []*domain.Task) error {
    taskIDs := make(map[string]bool, len(tasks))
    deps := make(map[string][]string, len(tasks))

    for _, task := range tasks {
        if task.ID == "" {
            return errors.New("task id is required")
        }
        if taskIDs[task.ID] {
            return errors.New("duplicate task ID: " + task.ID)
        }
        taskIDs[task.ID] = true
        deps[task.ID] = append([]string(nil), task.DependsOn...)
    }

    for taskID, ds := range deps {
        for _, depID := range ds {
            if depID == "" {
                return fmt.Errorf("task %s has empty dependency id", taskID)
            }
            if depID == taskID {
                return fmt.Errorf("task %s depends on itself", taskID)
            }
            if !taskIDs[depID] {
                return fmt.Errorf("task %s depends on unknown task %s", taskID, depID)
            }
        }
    }

    color := make(map[string]uint8, len(tasks))

    var visit func(string) error
    visit = func(n string) error {
        switch color[n] {
        case 1:
            return fmt.Errorf("cycle detected involving task %s", n)
        case 2:
            return nil
        }
        color[n] = 1
        for _, d := range deps[n] {
            if err := visit(d); err != nil {
                return err
            }
        }
        color[n] = 2
        return nil
    }

    for id := range deps {
        if color[id] == 0 {
            if err := visit(id); err != nil {
                return err
            }
        }
    }

    return nil
}

func (s *WorkflowService) ListWorkflows(ctx context.Context, limit int) ([]*domain.Workflow, error) {
    return s.workflowRepo.ListWorkflows(ctx, limit)
}