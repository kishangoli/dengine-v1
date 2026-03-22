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
    candidates := []string{"migrations"}

    if exe, err := os.Executable(); err == nil {
        exeDir := filepath.Dir(exe)
        candidates = append(candidates,
            filepath.Join(exeDir, "migrations"),
            filepath.Join(exeDir, "..", "..", "migrations"), 
        )
    }

    if wd, err := os.Getwd(); err == nil {
        for i := 0; i < 5; i++ {
            candidates = append(candidates, filepath.Join(wd, strings.Repeat("../", i), "migrations"))
        }
    }

    var migDir string
    for _, c := range candidates {
        clean := filepath.Clean(c)
        if st, err := os.Stat(clean); err == nil && st.IsDir() {
            migDir = clean
            break
        }
    }
    if migDir == "" {
        return fmt.Errorf("failed to locate migrations dir; tried: %v", candidates)
    }

    entries, err := os.ReadDir(migDir)
    if err != nil {
        return fmt.Errorf("failed to read migrations dir (%s): %w", migDir, err)
    }

    var files []string
    for _, e := range entries {
        if e.IsDir() {
            continue
        }
        name := e.Name()
        if strings.HasSuffix(name, ".sql") {
            files = append(files, filepath.Join(migDir, name))
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