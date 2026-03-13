package service

import (
    "context"
    "errors"
    "fmt"

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

    if err := s.workflowRepo.CreateWorkflow(ctx, workflow); err != nil {
        return fmt.Errorf("failed to store workflow: %w", err)
    }

    for _, task := range tasks {
        task.WorkflowID = workflow.ID

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
    taskIDs := make(map[string]bool)
    for _, task := range tasks {
        if taskIDs[task.ID] {
            return errors.New("duplicate task ID: " + task.ID)
        }
        taskIDs[task.ID] = true
    }

    // TODO: Add DAG cycle detection here (Step 5 will handle this in the scheduler).

    return nil
}