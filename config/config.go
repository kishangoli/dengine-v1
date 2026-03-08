package config

import (
    "os"
    "github.com/joho/godotenv"
)

type Config struct {
    DBPath       string
    APIPort      string
    OpenAIAPIKey string
}

func Load() (*Config, error) {
    _ = godotenv.Load()

    dbPath := os.Getenv("DB_PATH")
    if dbPath == "" {
        dbPath = "dengine.db"
    }

    apiPort := os.Getenv("API_PORT")
    if apiPort == "" {
        apiPort = "8080"
    }

    openAIKey := os.Getenv("OPENAI_API_KEY")

    return &Config{
        DBPath:       dbPath,
        APIPort:      apiPort,
        OpenAIAPIKey: openAIKey,
    }, nil
}