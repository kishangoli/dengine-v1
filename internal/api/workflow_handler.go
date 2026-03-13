package api

import (
    "encoding/json"
    "net/http"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/service"
)

type WorkflowHandler struct {
    workflowService *service.WorkflowService
}

func NewWorkflowHandler(workflowService *service.WorkflowService) *WorkflowHandler {
    return &WorkflowHandler{
        workflowService: workflowService,
    }
}

func (h *WorkflowHandler) SubmitWorkflow(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Workflow domain.Workflow   `json:"workflow"`
        Tasks    []*domain.Task    `json:"tasks"`
    }
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "invalid request body", http.StatusBadRequest)
        return
    }

    ctx := r.Context()
    if err := h.workflowService.SubmitWorkflow(ctx, &req.Workflow, req.Tasks); err != nil {
        http.Error(w, "failed to submit workflow: "+err.Error(), http.StatusInternalServerError)
        return
    }

    w.WriteHeader(http.StatusCreated)
}