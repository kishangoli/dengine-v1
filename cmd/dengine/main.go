package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"

    "github.com/kishangoli/dengine-v1/config"
    "github.com/kishangoli/dengine-v1/internal/api"
    "github.com/kishangoli/dengine-v1/internal/repository"
    "github.com/kishangoli/dengine-v1/internal/service"
)

func main() {
    cfg, err := config.Load()
    if err != nil {
        log.Fatalf("failed to load config: %v", err)
    }

    db, err := repository.InitDB(cfg.DBPath)
    if err != nil {
        log.Fatalf("failed to initialize database: %v", err)
    }
    defer db.Close()

    workflowRepo := repository.NewSQLiteWorkflowRepository(db)
    taskRepo := repository.NewSQLiteTaskRepository(db)
    dependencyRepo := repository.NewSQLiteDependencyRepository(db)

    workflowService := service.NewWorkflowService(workflowRepo, taskRepo, dependencyRepo)
    workflowHandler := api.NewWorkflowHandler(workflowService)

    scheduler := service.NewScheduler(taskRepo, dependencyRepo)

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    go scheduler.Start(ctx)

    http.HandleFunc("/workflows", workflowHandler.SubmitWorkflow)

    go func() {
        c := make(chan os.Signal, 1)
        signal.Notify(c, os.Interrupt, syscall.SIGTERM)
        <-c
        cancel()
    }()

    log.Printf("Starting server on port %s...", cfg.APIPort)
    log.Fatal(http.ListenAndServe(":"+cfg.APIPort, nil))
}