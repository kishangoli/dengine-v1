# Standalone C++ JSON process (Step 1)

This executable accepts one JSON request per input line and writes one JSON response per line. It stays running until stdin reaches EOF. At this stage it returns deterministic `echo: ` text; it does not load a model or call an API.

## Build

From the repository root, run:

```sh
make
```

This builds `native/dengine-native` in the project. Run `make test` to build if needed and check the protocol. Run `make clean` to remove only the generated executable. The executable is ignored by Git. On macOS, Xcode Command Line Tools provide the compiler and Make.

The pinned `nlohmann/json` v3.11.3 single header is vendored under `native/third_party/` with its MIT license. Builds and execution need no network access. The executable needs no model, API key, or GPU.

## Try it

```sh
printf '%s\n' \
  '{"id":"first","prompt":"hello"}' \
  '{"id":"second","prompt":"world"}' | ./native/dengine-native
```

The output is one response per request:

```json
{"id":"first","ok":true,"text":"echo: hello"}
{"id":"second","ok":true,"text":"echo: world"}
```

You can also run `./native/dengine-native` interactively, type one compact JSON request per line without surrounding shell quotes, and press Enter. The response appears immediately. Press Control-D to close stdin and exit. Run the `printf` example above at your shell prompt, not inside the running program.

## Protocol

- Requests are newline-delimited JSON objects with nonempty string fields `id` and `prompt`. Extra fields are ignored. Input lines may use LF or CRLF. A JSON-escaped `\n` inside a prompt is data, not a message boundary.
- A valid request returns `{"id":"...","ok":true,"text":"echo: ..."}`. It echoes the exact prompt, including leading/trailing spaces, escapes, and Unicode.
- Invalid requests return `{"id":...,"ok":false,"error":{"code":"...","message":"..."}}`. The ID is preserved if it was a nonblank string in an otherwise valid JSON object; otherwise it is `null`.
- Empty or whitespace-only lines are invalid JSON requests and receive `invalid_json`. A JSON value that is not an object receives `invalid_request`.
- A line longer than 65,536 bytes receives `request_too_large` with `id:null`. The process discards the rest of that line and continues with the next request. The limit includes any carriage return preceding LF, but excludes LF itself.
- A final input line without LF is processed on EOF. No response is generated for EOF with no pending input.
- Requests are processed sequentially. Stdout contains only response lines and is flushed after each response. Diagnostics use stderr. A failed read or write ends the process with a nonzero exit code.
- The response text is bounded in this step by the input-line limit plus the `echo: ` prefix. Step 2 will set a separate model output-token limit.

## Verify

```sh
make test
```

The checks cover multiple requests in one process, response flushing, recovery after invalid and oversized requests, escaping, CRLF, and an unterminated final line. They do not start the Go application or access its database.
