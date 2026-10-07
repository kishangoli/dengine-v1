"""Download exactly the selected public model revision outside Git tracking."""

from huggingface_hub import hf_hub_download

from common import MODEL_DIR, MODEL_INFO, sha256_file, verify_model_files


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in MODEL_INFO["files"].items():
        path = MODEL_DIR / name
        if path.is_file() and sha256_file(path) == expected_hash:
            print(f"verified {name}")
            continue
        print(f"downloading {name}", flush=True)
        hf_hub_download(
            repo_id=MODEL_INFO["model_id"],
            filename=name,
            revision=MODEL_INFO["revision"],
            local_dir=MODEL_DIR,
        )
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"downloaded file failed SHA-256 check: {name}")
    verify_model_files()
    print(f"model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
