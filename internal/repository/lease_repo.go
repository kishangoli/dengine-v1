package repository

import (
    "context"
    "database/sql"
    "fmt"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
)

type SQLiteLeaseRepository struct {
    db *sql.DB
}

func NewSQLiteLeaseRepository(db *sql.DB) LeaseRepository {
    return &SQLiteLeaseRepository{db: db}
}

func (r *SQLiteLeaseRepository) AcquireLease(ctx context.Context, lease *domain.Lease) (bool, error) {
    now := time.Now()

    // Try to take over an expired lease
    res, err := r.db.ExecContext(ctx,
        `UPDATE leases
         SET worker_id = ?, expires_at = ?, created_at = ?
         WHERE task_id = ? AND expires_at <= ?`,
        lease.WorkerID, lease.ExpiresAt, lease.CreatedAt,
        lease.TaskID, now,
    )
    if err != nil {
        return false, fmt.Errorf("failed to acquire lease (update expired): %w", err)
    }
    updated, err := res.RowsAffected()
    if err != nil {
        return false, fmt.Errorf("failed rows affected (update expired): %w", err)
    }
    if updated == 1 {
        return true, nil
    }

    // Otherwise try to insert a fresh lease
    res, err = r.db.ExecContext(ctx,
        `INSERT OR IGNORE INTO leases (task_id, worker_id, expires_at, created_at)
         VALUES (?, ?, ?, ?)`,
        lease.TaskID, lease.WorkerID, lease.ExpiresAt, lease.CreatedAt,
    )
    if err != nil {
        return false, fmt.Errorf("failed to acquire lease (insert): %w", err)
    }
    inserted, err := res.RowsAffected()
    if err != nil {
        return false, fmt.Errorf("failed rows affected (insert): %w", err)
    }

    return inserted == 1, nil
}

func (r *SQLiteLeaseRepository) RenewLease(ctx context.Context, taskID string, workerID string, newExpiry interface{}) error {
    result, err := r.db.ExecContext(ctx,
        "UPDATE leases SET expires_at = ? WHERE task_id = ? AND worker_id = ?",
        newExpiry, taskID, workerID,
    )
    if err != nil {
        return fmt.Errorf("failed to renew lease: %w", err)
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("failed to check rows affected: %w", err)
    }
    if rowsAffected == 0 {
        return fmt.Errorf("lease not found or owned by another worker")
    }

    return nil
}

func (r *SQLiteLeaseRepository) ReleaseLease(ctx context.Context, taskID string) error {
    _, err := r.db.ExecContext(ctx,
        "DELETE FROM leases WHERE task_id = ?", taskID,
    )
    if err != nil {
        return fmt.Errorf("failed to release lease: %w", err)
    }
    return nil
}

func (r *SQLiteLeaseRepository) GetExpiredLeases(ctx context.Context) ([]*domain.Lease, error) {
    rows, err := r.db.QueryContext(ctx,
        "SELECT task_id, worker_id, expires_at, created_at FROM leases WHERE expires_at < ?",
        time.Now(),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to get expired leases: %w", err)
    }
    defer rows.Close()

    var leases []*domain.Lease
    for rows.Next() {
        l := &domain.Lease{}
        if err := rows.Scan(&l.TaskID, &l.WorkerID, &l.ExpiresAt, &l.CreatedAt); err != nil {
            return nil, fmt.Errorf("failed to scan lease: %w", err)
        }
        leases = append(leases, l)
    }
    return leases, nil
}

func (r *SQLiteLeaseRepository) RecordHeartbeat(ctx context.Context, heartbeat *domain.WorkerHeartbeat) error {
    _, err := r.db.ExecContext(ctx,
        "INSERT OR REPLACE INTO worker_heartbeats (worker_id, task_id, timestamp) VALUES (?, ?, ?)",
        heartbeat.WorkerID, heartbeat.TaskID, heartbeat.Timestamp,
    )
    if err != nil {
        return fmt.Errorf("failed to record heartbeat: %w", err)
    }
    return nil
}