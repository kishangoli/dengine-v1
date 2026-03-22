package events

import (
    "sync"
)

type Bus struct {
    mu   sync.RWMutex
    subs map[string]map[chan Envelope]struct{} 
}

func NewBus() *Bus {
    return &Bus{
        subs: map[string]map[chan Envelope]struct{}{},
    }
}

func (b *Bus) Subscribe(workflowID string) (<-chan Envelope, func()) {
    ch := make(chan Envelope, 32)

    b.mu.Lock()
    if b.subs[workflowID] == nil {
        b.subs[workflowID] = map[chan Envelope]struct{}{}
    }
    b.subs[workflowID][ch] = struct{}{}
    b.mu.Unlock()

    unsub := func() {
        b.mu.Lock()
        if m := b.subs[workflowID]; m != nil {
            delete(m, ch)
            if len(m) == 0 {
                delete(b.subs, workflowID)
            }
        }
        b.mu.Unlock()
        close(ch)
    }

    return ch, unsub
}

func (b *Bus) Publish(env Envelope) {
    b.mu.RLock()
    m := b.subs[env.WorkflowID]
    b.mu.RUnlock()

    for ch := range m {
        select {
        case ch <- env:
        default:
        }
    }
}