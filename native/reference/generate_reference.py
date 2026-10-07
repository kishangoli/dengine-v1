"""Generate or verify CPU token/logit fixtures for the pinned model.

Python Transformers is a comparison implementation, not a runtime dependency
of the planned C++ inference engine.
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_DIR, MODEL_INFO, REFERENCE_DIR, verify_model_files


FIXTURE_DIR = REFERENCE_DIR / "fixtures"
GOLDEN_JSON = FIXTURE_DIR / "golden.json"
GOLDEN_LOGITS = FIXTURE_DIR / "logits.npz"
NEW_TOKENS = 8


def input_ids_for(case, tokenizer):
    if case["kind"] == "raw":
        return tokenizer(case["text"], return_tensors="pt").input_ids
    if case["kind"] == "chat":
        return tokenizer.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    raise ValueError(f"unknown fixture kind: {case['kind']}")


def calculate():
    verify_model_files()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    cases = json.loads((FIXTURE_DIR / "cases.json").read_text())
    output = {
        "model_id": MODEL_INFO["model_id"],
        "revision": MODEL_INFO["revision"],
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "reference_settings": {
            "device": "cpu",
            "compute_dtype": "float32",
            "attention": "eager",
            "generation": "greedy",
            "max_new_tokens": NEW_TOKENS,
            "threads": 4,
        },
        "cases": [],
    }
    logits = {}

    with torch.inference_mode():
        for case in cases:
            input_ids = input_ids_for(case, tokenizer)
            attention_mask = torch.ones_like(input_ids)
            scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, -1].float()
            scores_np = scores.cpu().numpy().copy()
            logits[case["name"]] = scores_np
            top = torch.topk(scores, 10)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )[0, input_ids.shape[1] :].tolist()
            output["cases"].append(
                {
                    "name": case["name"],
                    "input_ids": input_ids[0].tolist(),
                    "generated_ids": generated,
                    "generated_text": tokenizer.decode(
                        generated, skip_special_tokens=True
                    ),
                    "top10_next_token_ids": top.indices.tolist(),
                    "top10_next_token_logits": top.values.tolist(),
                }
            )
            print(f"computed {case['name']}", flush=True)
    return output, logits


def check(output, logits):
    expected = json.loads(GOLDEN_JSON.read_text())
    for key in (
        "model_id",
        "revision",
        "torch_version",
        "transformers_version",
        "reference_settings",
    ):
        if output[key] != expected[key]:
            raise AssertionError(f"reference metadata changed: {key}")
    if [c["name"] for c in output["cases"]] != [
        c["name"] for c in expected["cases"]
    ]:
        raise AssertionError("fixture cases changed")
    with np.load(GOLDEN_LOGITS) as expected_logits:
        for actual, saved in zip(output["cases"], expected["cases"]):
            for key in (
                "input_ids",
                "generated_ids",
                "generated_text",
                "top10_next_token_ids",
            ):
                if actual[key] != saved[key]:
                    raise AssertionError(f"{actual['name']}: {key} changed")
            if not np.allclose(
                logits[actual["name"]],
                expected_logits[actual["name"]],
                rtol=1e-4,
                atol=1e-4,
            ):
                raise AssertionError(f"{actual['name']}: next-token logits changed")
    print("all pinned reference fixtures match")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="compare against saved fixtures"
    )
    args = parser.parse_args()
    output, logits = calculate()
    if args.check:
        check(output, logits)
    else:
        GOLDEN_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        np.savez_compressed(GOLDEN_LOGITS, **logits)
        print(f"wrote {GOLDEN_JSON} and {GOLDEN_LOGITS}")


if __name__ == "__main__":
    main()
