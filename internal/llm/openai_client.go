package llm

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"
)

// OpenAI enforces a minimum max_output_tokens (currently 16).
const minMaxOutputTokens = 16

func clampMaxOutput(n int) int {
    if n <= 0 {
        return minMaxOutputTokens
    }
    if n < minMaxOutputTokens {
        return minMaxOutputTokens
    }
    return n
}

type Client interface {
    GenerateText(ctx context.Context, req GenerateTextRequest) (string, error)
}

type GenerateTextRequest struct {
    Model        string
    SystemPrompt string
    UserPrompt   string
    Temperature  float64
    MaxOutput    int
}

type OpenAIClient struct {
    apiKey     string
    httpClient *http.Client
    baseURL    string
}

func NewOpenAIClientFromEnv() (*OpenAIClient, error) {
    key := os.Getenv("OPENAI_API_KEY")
    if key == "" {
        return nil, fmt.Errorf("OPENAI_API_KEY is not set")
    }
    return NewOpenAIClient(key), nil
}

func NewOpenAIClient(apiKey string) *OpenAIClient {
    return &OpenAIClient{
        apiKey: apiKey,
        baseURL: "https://api.openai.com/v1",
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (c *OpenAIClient) GenerateText(ctx context.Context, req GenerateTextRequest) (string, error) {
    if req.Model == "" {
        req.Model = "gpt-4.1-mini"
    }
    if req.Temperature == 0 {
        req.Temperature = 0.2
    }

    req.MaxOutput = clampMaxOutput(req.MaxOutput)

    payload := map[string]any{
        "model": req.Model,
        "input": []map[string]any{
            {
                "role": "system",
                "content": []map[string]any{
                    {"type": "input_text", "text": req.SystemPrompt},
                },
            },
            {
                "role": "user",
                "content": []map[string]any{
                    {"type": "input_text", "text": req.UserPrompt},
                },
            },
        },
        "temperature": req.Temperature,
        "max_output_tokens": req.MaxOutput,
    }

    body, err := json.Marshal(payload)
    if err != nil {
        return "", fmt.Errorf("marshal openai request: %w", err)
    }

    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/responses", bytes.NewReader(body))
    if err != nil {
        return "", fmt.Errorf("create request: %w", err)
    }
    httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
    httpReq.Header.Set("Content-Type", "application/json")

    resp, err := c.httpClient.Do(httpReq)
    if err != nil {
        return "", fmt.Errorf("openai request failed: %w", err)
    }
    defer resp.Body.Close()

    respBody, _ := io.ReadAll(resp.Body)
    if resp.StatusCode < 200 || resp.StatusCode >= 300 {
        return "", fmt.Errorf("openai error status=%d body=%s", resp.StatusCode, string(respBody))
    }

    var decoded struct {
        Output []struct {
            Content []struct {
                Type string `json:"type"`
                Text string `json:"text"`
            } `json:"content"`
        } `json:"output"`
    }
    if err := json.Unmarshal(respBody, &decoded); err != nil {
        return "", fmt.Errorf("unmarshal openai response: %w", err)
    }

    var out bytes.Buffer
    for _, o := range decoded.Output {
        for _, c := range o.Content {
            if c.Type == "output_text" || c.Type == "text" {
                out.WriteString(c.Text)
            }
        }
    }
    if out.Len() == 0 {
        return "", fmt.Errorf("openai response had no text output")
    }
    return out.String(), nil
}