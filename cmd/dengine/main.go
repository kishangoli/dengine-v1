package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/kishangoli/dengine-v1/config"
    "github.com/kishangoli/dengine-v1/internal/api"
    "github.com/kishangoli/dengine-v1/internal/executor"
    "github.com/kishangoli/dengine-v1/internal/llm"
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
    leaseRepo := repository.NewSQLiteLeaseRepository(db)

    workflowService := service.NewWorkflowService(workflowRepo, taskRepo, dependencyRepo)
    workflowHandler := api.NewWorkflowHandler(workflowService)

    scheduler := service.NewScheduler(taskRepo, dependencyRepo)

    openaiClient, err := llm.NewOpenAIClientFromEnv()
    if err != nil {
        log.Fatalf("failed to init openai client: %v", err)
    }

    execRegistry := executor.NewRegistry()
    llmExec := executor.NewLLMExecutor(openaiClient, "gpt-4.1-mini")
    execRegistry.Register("extract", llmExec)
    execRegistry.Register("summarize", llmExec)
    execRegistry.Register("classify", llmExec)

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    go scheduler.Start(ctx)

    for i := 0; i < 5; i++ {
        w := service.NewWorker(fmt.Sprintf("worker-%d", i+1), taskRepo, leaseRepo, execRegistry)
        go w.Start(ctx)
    }

    mux := http.NewServeMux()
    mux.HandleFunc("/workflows", workflowHandler.SubmitWorkflow)

    srv := &http.Server{
        Addr:    ":" + cfg.APIPort,
        Handler: mux,
    }

    go func() {
        c := make(chan os.Signal, 1)
        signal.Notify(c, os.Interrupt, syscall.SIGTERM)
        <-c

        cancel()

        shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer shutdownCancel()
        _ = srv.Shutdown(shutdownCtx)
    }()

    log.Printf("Starting server on port %s...", cfg.APIPort)
    if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
        log.Fatal(err)
    }
}