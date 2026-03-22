package repository

import (
    "database/sql"
    "fmt"
    "os"
    "path/filepath"
    "sort"
    "strings"

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
    entries, err := os.ReadDir("migrations")
    if err != nil {
        return fmt.Errorf("failed to read migrations dir: %w", err)
    }

    var files []string
    for _, e := range entries {
        if e.IsDir() {
            continue
        }
        name := e.Name()
        if strings.HasSuffix(name, ".sql") {
            files = append(files, filepath.Join("migrations", name))
        }
    }

    sort.Strings(files)
    for _, f := range files {
        migrationSQL, err := os.ReadFile(f)
        if err != nil {
            return fmt.Errorf("failed to read migration file %s: %w", f, err)
        }
        if _, err := db.Exec(string(migrationSQL)); err != nil {
            return fmt.Errorf("failed to execute migration %s: %w", f, err)
        }
    }

    return nil
}