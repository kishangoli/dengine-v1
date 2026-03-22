package api

import (
    "encoding/json"
    "net/http"
    "time"

    "github.com/gorilla/websocket"
    "github.com/kishangoli/dengine-v1/internal/events"
)

type WSHandler struct {
    bus *events.Bus
}

func NewWSHandler(bus *events.Bus) *WSHandler {
    return &WSHandler{bus: bus}
}

type wsClientMsg struct {
    Type       string `json:"type"`
    WorkflowID string `json:"workflow_id"`
}

var upgrader = websocket.Upgrader{
    CheckOrigin: func(r *http.Request) bool { return true }, // localhost dev
}

func (h *WSHandler) ServeWS(w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        return
    }
    defer conn.Close()

    _, b, err := conn.ReadMessage()
    if err != nil {
        return
    }
    var msg wsClientMsg
    if err := json.Unmarshal(b, &msg); err != nil || msg.Type != "subscribe" || msg.WorkflowID == "" {
        _ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"error","message":"expected subscribe"}`))
        return
    }

    eventsCh, unsub := h.bus.Subscribe(msg.WorkflowID)
    defer unsub()

    _ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
    conn.SetPongHandler(func(string) error {
        _ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
        return nil
    })

    ticker := time.NewTicker(20 * time.Second)
    defer ticker.Stop()

    done := make(chan struct{})
    go func() {
        defer close(done)
        for {
            if _, _, err := conn.ReadMessage(); err != nil {
                return
            }
        }
    }()

    for {
        select {
        case <-r.Context().Done():
            return
        case <-done:
            return
        case <-ticker.C:
            _ = conn.WriteMessage(websocket.PingMessage, []byte("ping"))
        case env := <-eventsCh:
            out, _ := json.Marshal(env)
            if err := conn.WriteMessage(websocket.TextMessage, out); err != nil {
                return
            }
        }
    }
}