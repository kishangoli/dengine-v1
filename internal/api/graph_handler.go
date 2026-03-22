package api

import (
    "encoding/json"
    "net/http"
    "strings"

    "github.com/kishangoli/dengine-v1/internal/repository"
)

type Edge struct {
    From string `json:"from"`
    To   string `json:"to"`
}

type GraphResponse struct {
    WorkflowID   string `json:"workflow_id"`
    WorkflowName string `json:"workflow_name"`
    Tasks        any    `json:"tasks"`
    Edges        []Edge `json:"edges"`
}

type GraphHandler struct {
    workflowRepo   repository.WorkflowRepository
    taskRepo       repository.TaskRepository
    dependencyRepo repository.DependencyRepository
}

func NewGraphHandler(
    workflowRepo repository.WorkflowRepository,
    taskRepo repository.TaskRepository,
    dependencyRepo repository.DependencyRepository,
) *GraphHandler {
    return &GraphHandler{
        workflowRepo:   workflowRepo,
        taskRepo:       taskRepo,
        dependencyRepo: dependencyRepo,
    }
}

func (h *GraphHandler) GetWorkflowGraph(w http.ResponseWriter, r *http.Request) {
    path := strings.TrimPrefix(r.URL.Path, "/api/workflows/")
    path = strings.TrimSuffix(path, "/graph")
    workflowID := strings.Trim(path, "/")
    if workflowID == "" {
        http.Error(w, "missing workflow id", http.StatusBadRequest)
        return
    }

    ctx := r.Context()

    wf, err := h.workflowRepo.GetWorkflow(ctx, workflowID)
    if err != nil {
        http.Error(w, "failed to load workflow: "+err.Error(), http.StatusInternalServerError)
        return
    }

    tasks, err := h.taskRepo.GetTasksByWorkflow(ctx, workflowID)
    if err != nil {
        http.Error(w, "failed to load tasks: "+err.Error(), http.StatusInternalServerError)
        return
    }

    deps, err := h.dependencyRepo.GetDependenciesByWorkflow(ctx, workflowID)
    if err != nil {
        http.Error(w, "failed to load dependencies: "+err.Error(), http.StatusInternalServerError)
        return
    }

    edges := make([]Edge, 0, len(deps))
    for _, d := range deps {
        edges = append(edges, Edge{From: d.DependsOnID, To: d.TaskID})
    }

    resp := GraphResponse{
        WorkflowID:   workflowID,
        WorkflowName: wf.Name,
        Tasks:        tasks,
        Edges:        edges,
    }

    w.Header().Set("Content-Type", "application/json")
    _ = json.NewEncoder(w).Encode(resp)
}