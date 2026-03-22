FROM golang:1.23-bookworm AS build
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -o /out/dengine ./cmd/dengine

FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates sqlite3 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/dengine /app/dengine
COPY migrations /app/migrations

ENV API_PORT=8080
ENV DB_PATH=/tmp/dengine.db
EXPOSE 8080

CMD ["/app/dengine"]