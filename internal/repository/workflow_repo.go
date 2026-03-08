package repository

import (
    "context"
    "database/sql"
    "fmt"
    "time"
    "github.com/kishangoli/dengine-v1/internal/domain"
)

type SQLiteWorkflowRepository struct {
    db *sql.DB
}

func NewSQLiteWorkflowRepository(db *sql.DB) WorkflowRepository {
    return &SQLiteWorkflowRepository{db: db}
}

func (r *SQLiteWorkflowRepository) CreateWorkflow(ctx context.Context, workflow *domain.Workflow) error {
    _, err := r.db.ExecContext(ctx,
        "INSERT INTO workflows (id, name, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        workflow.ID, workflow.Name, workflow.Status, workflow.CreatedAt, workflow.UpdatedAt,
    )
    if err != nil {
        return fmt.Errorf("failed to create workflow: %w", err)
    }
    return nil
}

func (r *SQLiteWorkflowRepository) GetWorkflow(ctx context.Context, id string) (*domain.Workflow, error) {
    w := &domain.Workflow{}
    err := r.db.QueryRowContext(ctx,
        "SELECT id, name, status, created_at, updated_at FROM workflows WHERE id = ?", id,
    ).Scan(&w.ID, &w.Name, &w.Status, &w.CreatedAt, &w.UpdatedAt)
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("workflow not found: %s", id)
    }
    if err != nil {
        return nil, fmt.Errorf("failed to get workflow: %w", err)
    }
    return w, nil
}

func (r *SQLiteWorkflowRepository) UpdateWorkflowStatus(ctx context.Context, id string, status domain.Status) error {
    _, err := r.db.ExecContext(ctx,
        "UPDATE workflows SET status = ?, updated_at = ? WHERE id = ?",
        status, time.Now(), id,
    )
    if err != nil {
        return fmt.Errorf("failed to update workflow status: %w", err)
    }
    return nil
}