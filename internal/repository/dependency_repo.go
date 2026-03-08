package repository

import (
    "context"
    "database/sql"
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
    var count int
    err := r.db.QueryRowContext(ctx,
        `SELECT COUNT(*) FROM task_dependencies td
         JOIN tasks t ON td.depends_on_id = t.id
         WHERE td.task_id = ? AND t.status != ?`,
        taskID, domain.StatusCompleted,
    ).Scan(&count)
    if err != nil {
        return false, fmt.Errorf("failed to check dependencies: %w", err)
    }
    return count == 0, nil
}