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
    var lastErr sql.NullString
    var nextRunAt sql.NullTime

    err := r.db.QueryRowContext(ctx,
        `SELECT id, workflow_id, type, input, status, output,
                retry_count, max_retries, timeout_ns,
                attempts, max_attempts, last_error, next_run_at,
                created_at, updated_at
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
        &t.Attempts,
        &t.MaxAttempts,
        &lastErr,
        &nextRunAt,
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
    if lastErr.Valid {
        t.LastError = &lastErr.String
    } else {
        t.LastError = nil
    }
    if nextRunAt.Valid {
        tr := nextRunAt.Time
        t.NextRunAt = &tr
    } else {
        t.NextRunAt = nil
    }

    t.Timeout = time.Duration(timeoutNS)
    return t, nil
}

func (r *SQLiteTaskRepository) GetTasksByWorkflow(ctx context.Context, workflowID string) ([]*domain.Task, error) {
    rows, err := r.db.QueryContext(ctx,
        `SELECT id, workflow_id, type, input, status, output,
                retry_count, max_retries, timeout_ns,
                attempts, max_attempts, last_error, next_run_at,
                created_at, updated_at
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
        var lastErr sql.NullString
        var nextRunAt sql.NullTime

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
            &t.Attempts,
            &t.MaxAttempts,
            &lastErr,
            &nextRunAt,
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
        if lastErr.Valid {
            t.LastError = &lastErr.String
        }
        if nextRunAt.Valid {
            tr := nextRunAt.Time
            t.NextRunAt = &tr
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
    now := time.Now()

    rows, err := r.db.QueryContext(ctx,
        `SELECT id, workflow_id, type, input, status, output,
                retry_count, max_retries, timeout_ns,
                attempts, max_attempts, last_error, next_run_at,
                created_at, updated_at
         FROM tasks
         WHERE status = ?
           AND (next_run_at IS NULL OR next_run_at <= ?)`,
        status, now,
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
        var lastErr sql.NullString
        var nextRunAt sql.NullTime

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
            &t.Attempts,
            &t.MaxAttempts,
            &lastErr,
            &nextRunAt,
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
        if lastErr.Valid {
            t.LastError = &lastErr.String
        }
        if nextRunAt.Valid {
            tr := nextRunAt.Time
            t.NextRunAt = &tr
        }

        t.Timeout = time.Duration(timeoutNS)
        tasks = append(tasks, t)
    }
    return tasks, nil
}

func (r *SQLiteTaskRepository) IncrementTaskAttempts(ctx context.Context, taskID string) error {
    _, err := r.db.ExecContext(ctx, `
        UPDATE tasks
        SET attempts = attempts + 1,
            updated_at = ?
        WHERE id = ?`,
        time.Now(), taskID,
    )
    if err != nil {
        return fmt.Errorf("increment attempts: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) SetTaskError(ctx context.Context, taskID string, msg string) error {
    _, err := r.db.ExecContext(ctx, `
        UPDATE tasks
        SET last_error = ?,
            updated_at = ?
        WHERE id = ?`,
        msg, time.Now(), taskID,
    )
    if err != nil {
        return fmt.Errorf("set last_error: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) ClearTaskError(ctx context.Context, taskID string) error {
    _, err := r.db.ExecContext(ctx, `
        UPDATE tasks
        SET last_error = NULL,
            updated_at = ?
        WHERE id = ?`,
        time.Now(), taskID,
    )
    if err != nil {
        return fmt.Errorf("clear last_error: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) SetTaskNextRunAt(ctx context.Context, taskID string, t *time.Time) error {
    var next any
    if t == nil {
        next = nil
    } else {
        next = *t
    }
    _, err := r.db.ExecContext(ctx, `
        UPDATE tasks
        SET next_run_at = ?,
            updated_at = ?
        WHERE id = ?`,
        next, time.Now(), taskID,
    )
    if err != nil {
        return fmt.Errorf("set next_run_at: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) MarkTaskRunnable(ctx context.Context, taskID string) error {
    _, err := r.db.ExecContext(ctx, `
        UPDATE tasks
        SET status = ?,
            updated_at = ?
        WHERE id = ?`,
        domain.StatusRunnable, time.Now(), taskID,
    )
    if err != nil {
        return fmt.Errorf("mark runnable: %w", err)
    }
    return nil
}

func (r *SQLiteTaskRepository) GetStaleRunningTasks(ctx context.Context, now time.Time) ([]*domain.Task, error) {
    // stale = running AND (no lease row OR lease.expires_at <= now)
    rows, err := r.db.QueryContext(ctx, `
        SELECT t.id, t.workflow_id, t.type, t.status, t.input, t.output,
               t.attempts, t.max_attempts, t.last_error, t.next_run_at,
               t.created_at, t.updated_at
        FROM tasks t
        LEFT JOIN leases l ON l.task_id = t.id
        WHERE t.status = ?
          AND (l.task_id IS NULL OR l.expires_at <= ?)`,
        domain.StatusRunning, now,
    )
    if err != nil {
        return nil, fmt.Errorf("get stale running tasks: %w", err)
    }
    defer rows.Close()

    var tasks []*domain.Task
    for rows.Next() {
        var (
            t         domain.Task
            lastErr   sql.NullString
            nextRunAt sql.NullTime
        )

        if err := rows.Scan(
            &t.ID, &t.WorkflowID, &t.Type, &t.Status, &t.Input, &t.Output,
            &t.Attempts, &t.MaxAttempts, &lastErr, &nextRunAt,
            &t.CreatedAt, &t.UpdatedAt,
        ); err != nil {
            return nil, fmt.Errorf("scan task: %w", err)
        }
        if lastErr.Valid {
            t.LastError = &lastErr.String
        }
        if nextRunAt.Valid {
            tr := nextRunAt.Time
            t.NextRunAt = &tr
        }

        tasks = append(tasks, &t)
    }

    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("iterate tasks: %w", err)
    }

    return tasks, nil
}