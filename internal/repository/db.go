package repository

import (
    "database/sql"
    "fmt"
    "os"
    _ "github.com/mattn/go-sqlite3"
)

func InitDB(dbPath string) (*sql.DB, error) {
    db, err := sql.Open("sqlite3", dbPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }

    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }

    _, err = db.Exec("PRAGMA foreign_keys = ON")
    if err != nil {
        return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
    }

    if err := runMigrations(db); err != nil {
        return nil, fmt.Errorf("failed to run migrations: %w", err)
    }

    return db, nil
}

func runMigrations(db *sql.DB) error {
    migrationSQL, err := os.ReadFile("migrations/001_init.sql")
    if err != nil {
        return fmt.Errorf("failed to read migration file: %w", err)
    }

    _, err = db.Exec(string(migrationSQL))
    if err != nil {
        return fmt.Errorf("failed to execute migration: %w", err)
    }

    return nil
}