"""Save selected outputs from one pretrained transformer layer for C++ checks."""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from common import MODEL_DIR, REFERENCE_DIR, verify_model_files


FIXTURE = REFERENCE_DIR / "fixtures" / "block_layer0.json"
CASES = [
    ("single", [19556], 0),
    ("four_tokens", [19556, 28, 905, 17], 0),
    ("four_tokens_offset", [19556, 28, 905, 17], 11),
]


def calculate():
    verify_model_files()
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval().model
    layer = model.layers[0]
    result = {"layer": 0, "cases": []}
    with torch.inference_mode():
        for name, ids, start in CASES:
            hidden = model.embed_tokens(torch.tensor([ids]))
            positions = torch.arange(start, start + len(ids)).unsqueeze(0)
            embeddings = model.rotary_emb(hidden, positions)
            mask = torch.full((1, 1, len(ids), len(ids)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            normalized = layer.input_layernorm(hidden)
            attention_output, probabilities = layer.self_attn(
                normalized, position_embeddings=embeddings,
                attention_mask=mask, use_cache=False,
            )
            output = layer(
                hidden, position_embeddings=embeddings,
                attention_mask=mask, use_cache=False,
            )
            result["cases"].append({
                "name": name,
                "input_ids": ids,
                "start_position": start,
                "normalized_first_8": normalized[0, 0, :8].tolist(),
                "attention_output_first_8": attention_output[0, 0, :8].tolist(),
                "attention_head0_last": probabilities[0, 0, -1].tolist(),
                "output": output[0].tolist(),
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    actual = calculate()
    if args.check:
        expected = json.loads(FIXTURE.read_text())
        if actual["layer"] != expected["layer"] or len(actual["cases"]) != len(expected["cases"]):
            raise AssertionError("block reference metadata changed")
        for current, saved in zip(actual["cases"], expected["cases"]):
            for key in ("name", "input_ids", "start_position"):
                if current[key] != saved[key]:
                    raise AssertionError(f"block reference metadata changed: {key}")
            for key in ("normalized_first_8", "attention_output_first_8",
                        "attention_head0_last", "output"):
                if not np.allclose(current[key], saved[key], rtol=1e-5, atol=1e-5):
                    raise AssertionError(f"block reference changed: {current['name']} {key}")
        print("block reference fixture matches")
    else:
        FIXTURE.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
