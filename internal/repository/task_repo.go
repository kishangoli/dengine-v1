package repository

import (
    "context"
    "database/sql"
    "fmt"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
)

type SQLiteTaskRepository struct {
    db *sql.DB
}

func NewSQLiteTaskRepository(db *sql.DB) TaskRepository {
    return &SQLiteTaskRepository{db: db}
}

func (r *SQLiteTaskRepository) CreateTask(ctx context.Context, task *domain.Task) error {
    _, err := r.db.ExecContext(ctx,
        `INSERT INTO tasks (id, workflow_id, type, input, status, retry_count, max_retries, timeout_ns, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        task.ID, task.WorkflowID, task.Type, task.Input, task.Status,
        task.RetryCount, task.MaxRetries, task.Timeout.Nanoseconds(),
        task.CreatedAt, task.UpdatedAt,
    )
    if err != nil {
        return fmt.Errorf("failed to create task: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) GetTask(ctx context.Context, id string) (*domain.Task, error) {
    t := &domain.Task{}
    var timeoutNS int64
    var input sql.NullString
    var output sql.NullString

    err := r.db.QueryRowContext(ctx,
        `SELECT id, workflow_id, type, input, status, output, retry_count, max_retries, timeout_ns, created_at, updated_at
         FROM tasks WHERE id = ?`, id,
    ).Scan(
        &t.ID,
        &t.WorkflowID,
        &t.Type,
        &input,
        &t.Status,
        &output,
        &t.RetryCount,
        &t.MaxRetries,
        &timeoutNS,
        &t.CreatedAt,
        &t.UpdatedAt,
    )

    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("task not found: %s", id)
    }
    if err != nil {
        return nil, fmt.Errorf("failed to get task: %w", err)
    }

    if input.Valid {
        t.Input = input.String
    } else {
        t.Input = ""
    }
    if output.Valid {
        t.Output = output.String
    } else {
        t.Output = ""
    }

    t.Timeout = time.Duration(timeoutNS)
    return t, nil
}

func (r *SQLiteTaskRepository) GetTasksByWorkflow(ctx context.Context, workflowID string) ([]*domain.Task, error) {
    rows, err := r.db.QueryContext(ctx,
        `SELECT id, workflow_id, type, input, status, output, retry_count, max_retries, timeout_ns, created_at, updated_at
         FROM tasks WHERE workflow_id = ?`, workflowID,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to get tasks: %w", err)
    }
    defer rows.Close()

    var tasks []*domain.Task
    for rows.Next() {
        t := &domain.Task{}
        var timeoutNS int64
        var input sql.NullString
        var output sql.NullString

        err := rows.Scan(
            &t.ID,
            &t.WorkflowID,
            &t.Type,
            &input,
            &t.Status,
            &output,
            &t.RetryCount,
            &t.MaxRetries,
            &timeoutNS,
            &t.CreatedAt,
            &t.UpdatedAt,
        )
        if err != nil {
            return nil, fmt.Errorf("failed to scan task: %w", err)
        }

        if input.Valid {
            t.Input = input.String
        } else {
            t.Input = ""
        }
        if output.Valid {
            t.Output = output.String
        } else {
            t.Output = ""
        }

        t.Timeout = time.Duration(timeoutNS)
        tasks = append(tasks, t)
    }
    return tasks, nil
}

func (r *SQLiteTaskRepository) UpdateTaskStatus(ctx context.Context, id string, status domain.Status) error {
    _, err := r.db.ExecContext(ctx,
        "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
        status, time.Now(), id,
    )
    if err != nil {
        return fmt.Errorf("failed to update task status: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) UpdateTaskOutput(ctx context.Context, id string, output string) error {
    _, err := r.db.ExecContext(ctx,
        "UPDATE tasks SET output = ?, updated_at = ? WHERE id = ?",
        output, time.Now(), id,
    )
    if err != nil {
        return fmt.Errorf("failed to update task output: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) IncrementRetryCount(ctx context.Context, id string) error {
    _, err := r.db.ExecContext(ctx,
        "UPDATE tasks SET retry_count = retry_count + 1, updated_at = ? WHERE id = ?",
        time.Now(), id,
    )
    if err != nil {
        return fmt.Errorf("failed to increment retry count: %w", err)
    }
    return nil
}


func (r *SQLiteTaskRepository) GetPendingTasks(ctx context.Context) ([]*domain.Task, error) {
    return r.getTasksByStatus(ctx, domain.StatusPending)
}

func (r *SQLiteTaskRepository) GetRunnableTasks(ctx context.Context) ([]*domain.Task, error) {
    return r.getTasksByStatus(ctx, domain.StatusRunnable)
}

// helper to avoid duplicating scan logic
func (r *SQLiteTaskRepository) getTasksByStatus(ctx context.Context, status domain.Status) ([]*domain.Task, error) {
    rows, err := r.db.QueryContext(ctx,
        `SELECT id, workflow_id, type, input, status, output, retry_count, max_retries, timeout_ns, created_at, updated_at
         FROM tasks WHERE status = ?`,
        status,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to query tasks by status %s: %w", status, err)
    }
    defer rows.Close()

    var tasks []*domain.Task
    for rows.Next() {
        t := &domain.Task{}
        var timeoutNS int64
        var input sql.NullString
        var output sql.NullString

        err := rows.Scan(
            &t.ID,
            &t.WorkflowID,
            &t.Type,
            &input,
            &t.Status,
            &output,
            &t.RetryCount,
            &t.MaxRetries,
            &timeoutNS,
            &t.CreatedAt,
            &t.UpdatedAt,
        )
        if err != nil {
            return nil, fmt.Errorf("failed to scan task: %w", err)
        }

        if input.Valid {
            t.Input = input.String
        } else {
            t.Input = ""
        }
        if output.Valid {
            t.Output = output.String
        } else {
            t.Output = ""
        }

        t.Timeout = time.Duration(timeoutNS)
        tasks = append(tasks, t)
    }
    return tasks, nil
}