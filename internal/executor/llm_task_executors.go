package executor

import (
    "context"
    "fmt"
    "strings"
    "time"

    "github.com/kishangoli/dengine-v1/internal/domain"
    "github.com/kishangoli/dengine-v1/internal/llm"
)

type LLMExecutor struct {
    client llm.Client
    model  string
}

func NewLLMExecutor(client llm.Client, model string) *LLMExecutor {
    if model == "" {
        model = "gpt-4.1-mini"
    }
    return &LLMExecutor{client: client, model: model}
}

func (e *LLMExecutor) Execute(ctx context.Context, task *domain.Task) (*Result, error) {
    input := strings.TrimSpace(task.Input)

    var (
        text string
        err  error
    )

    switch task.Type {
    case "extract":
        text, err = e.extract(ctx, input)
    case "summarize":
        text, err = e.summarize(ctx, input)
    case "classify":
        text, err = e.classify(ctx, input)
    default:
        return nil, fmt.Errorf("LLMExecutor: unknown task type %q", task.Type)
    }
    if err != nil {
        return nil, err
    }

    return &Result{
        TaskID:      task.ID,
        TaskType:    task.Type,
        Model:       e.model,
        Text:        text,
        GeneratedAt: time.Now(),
    }, nil
}

func (e *LLMExecutor) extract(ctx context.Context, input string) (string, error) {
    return e.client.GenerateText(ctx, llm.GenerateTextRequest{
        Model:        e.model,
        SystemPrompt: "You extract key facts from raw text. Output bullet points only.",
        UserPrompt:   input,
        Temperature:  0.1,
        MaxOutput:    120,
    })
}

func (e *LLMExecutor) summarize(ctx context.Context, input string) (string, error) {
    return e.client.GenerateText(ctx, llm.GenerateTextRequest{
        Model:        e.model,
        SystemPrompt: "You write concise summaries. Output 3-6 sentences.",
        UserPrompt:   input,
        Temperature:  0.3,
        MaxOutput:    120,
    })
}

func (e *LLMExecutor) classify(ctx context.Context, input string) (string, error) {
    return e.client.GenerateText(ctx, llm.GenerateTextRequest{
        Model:        e.model,
        SystemPrompt: "Classify the text into a single short label. Output only the label.",
        UserPrompt:   input,
        Temperature:  0,
        MaxOutput:    16,
    })
}