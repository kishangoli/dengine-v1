package repository

import (
    "context"
    "database/sql"
    "log"
    "fmt"

    "github.com/kishangoli/dengine-v1/internal/domain"
)

type SQLiteDependencyRepository struct {
    db *sql.DB
}

func NewSQLiteDependencyRepository(db *sql.DB) DependencyRepository {
    return &SQLiteDependencyRepository{db: db}
}

func (r *SQLiteDependencyRepository) CreateDependency(ctx context.Context, dep *domain.TaskDependency) error {
    _, err := r.db.ExecContext(ctx,
        "INSERT INTO task_dependencies (task_id, depends_on_id) VALUES (?, ?)",
        dep.TaskID, dep.DependsOnID,
    )
    if err != nil {
        return fmt.Errorf("failed to create dependency: %w", err)
    }
    return nil
}

func (r *SQLiteDependencyRepository) GetDependencies(ctx context.Context, taskID string) ([]*domain.TaskDependency, error) {
    rows, err := r.db.QueryContext(ctx,
        "SELECT task_id, depends_on_id FROM task_dependencies WHERE task_id = ?", taskID,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to get dependencies: %w", err)
    }
    defer rows.Close()

    var deps []*domain.TaskDependency
    for rows.Next() {
        d := &domain.TaskDependency{}
        if err := rows.Scan(&d.TaskID, &d.DependsOnID); err != nil {
            return nil, fmt.Errorf("failed to scan dependency: %w", err)
        }
        deps = append(deps, d)
    }
    return deps, nil
}

func (r *SQLiteDependencyRepository) AreDependenciesMet(ctx context.Context, taskID string) (bool, error) {
    log.Printf("Checking dependencies for task %s", taskID)
    var count int
    err := r.db.QueryRowContext(ctx,
        `SELECT COUNT(*) FROM task_dependencies td
         JOIN tasks t ON td.depends_on_id = t.id
         WHERE td.task_id = ? AND t.status != ?`,
        taskID, domain.StatusCompleted,
    ).Scan(&count)
    if err != nil {
        log.Printf("Error checking dependencies for task %s: %v", taskID, err)
        return false, fmt.Errorf("failed to check dependencies: %w", err)
    }

    log.Printf("Task %s has %d unmet dependencies", taskID, count)
    return count == 0, nil
}

func (r *SQLiteDependencyRepository) GetDependenciesByWorkflow(ctx context.Context, workflowID string) ([]*domain.TaskDependency, error) {
    rows, err := r.db.QueryContext(ctx, `
        SELECT td.task_id, td.depends_on_id
        FROM task_dependencies td
        JOIN tasks t ON t.id = td.task_id
        WHERE t.workflow_id = ?`, workflowID)
    if err != nil {
        return nil, fmt.Errorf("get dependencies by workflow: %w", err)
    }
    defer rows.Close()

    var out []*domain.TaskDependency
    for rows.Next() {
        d := &domain.TaskDependency{}
        if err := rows.Scan(&d.TaskID, &d.DependsOnID); err != nil {
            return nil, fmt.Errorf("scan dependency: %w", err)
        }
        out = append(out, d)
    }
    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("iterate dependencies: %w", err)
    }
    return out, nil
}