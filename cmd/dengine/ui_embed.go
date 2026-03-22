package main

import (
    "embed"
    "io/fs"
    "net/http"
)

//go:embed web/ui/*
var uiFS embed.FS

func uiHandler() (http.Handler, error) {
    sub, err := fs.Sub(uiFS, "web/ui")
    if err != nil {
        return nil, err
    }
    return http.StripPrefix("/ui/", http.FileServer(http.FS(sub))), nil
}