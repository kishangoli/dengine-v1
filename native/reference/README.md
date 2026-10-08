# Stage 1: SmolLM2 reference

This folder contains the **comparison implementation** for the future C++ engine. Python Transformers runs the selected pretrained model and saves its token IDs and next-token scores. The C++ engine will implement those steps itself; the Python packages are for checking our work, not part of the C++ runtime.

See [MODEL.md](MODEL.md) for a beginner-friendly description of the chosen model and its weight layout.

## Files and folders

- `model_info.json` pins the model revision, license, weight format, and SHA-256 of every required file.
- `download_model.py` downloads only those files into `native/models/smollm2-135m-instruct/`. That model folder is ignored by Git.
- `fixtures/cases.json` contains four fixed prompts: punctuation, whitespace, Unicode, and a formatted chat request.
- `generate_reference.py` runs the model on CPU. It stores input and generated token IDs in `fixtures/golden.json` and full next-token score arrays in `fixtures/logits.npz`.
- `test_reference.py` verifies file integrity, expected model structure, fixture coverage, and that a fresh Python run matches the saved results.
- `requirements.lock.txt` records the exact installed Python package versions used to produce the fixtures.

## Reproduce on the M4 Mac

Run from the repository root. Python 3.12 is used because the system's Python 3.14 does not have these ML packages installed.

```sh
python3.12 -m venv native/reference/.venv
native/reference/.venv/bin/python -m pip install -r native/reference/requirements.lock.txt
native/reference/.venv/bin/python native/reference/download_model.py
native/reference/.venv/bin/python native/reference/generate_reference.py --check
native/reference/.venv/bin/python -m unittest discover -s native/reference -p 'test_reference.py'
```

The download requires internet access once. Subsequent model runs use local files only. The model download is approximately 269 MB; Python packages and full-precision execution use additional disk space and memory. No API key or dedicated GPU is needed.

If the fixture inputs or reference settings intentionally change, run `generate_reference.py` without `--check`, inspect the resulting diff, and run the tests again. Keep model files out of Git; commit only the small reference fixtures and scripts.

## What the files mean

`golden.json` is readable: it records the exact prompt token IDs, eight greedily generated token IDs, a short text preview, and the ten highest next-token scores. `logits.npz` holds the **entire** next-token score vector for each prompt in a compact NumPy file. Comparing the full vector within a numeric tolerance will help detect math mistakes in C++ even when the top generated word looks plausible. The Unicode raw prompt may produce awkward text: this fixture tests tokenization and numerical agreement, not answer quality.

The reference loads the source BF16 weights and converts computation to float32 on CPU. This simplifies the first C++ correctness target; conversion does not add information to the original weights. Generation uses greedy selection with no randomness. Its outputs should be compared as token IDs and numeric scores before judging the text.
