"""Integration checks for the pinned model and saved Python reference."""

import json
import subprocess
import sys
import unittest

import numpy as np
import torch
from safetensors import safe_open

from common import MODEL_DIR, MODEL_INFO, REFERENCE_DIR, verify_model_files


class StageOneReferenceTests(unittest.TestCase):
    def test_pinned_files_and_model_layout(self):
        verify_model_files()
        config = json.loads((MODEL_DIR / "config.json").read_text())
        self.assertEqual(config["architectures"], ["LlamaForCausalLM"])
        self.assertEqual(config["num_hidden_layers"], 30)
        self.assertEqual(config["hidden_size"], 576)
        self.assertEqual(config["num_attention_heads"], 9)
        self.assertEqual(config["num_key_value_heads"], 3)
        self.assertEqual(config["vocab_size"], 49152)
        self.assertTrue(config["tie_word_embeddings"])

        with safe_open(MODEL_DIR / "model.safetensors", framework="pt", device="cpu") as weights:
            self.assertEqual(len(weights.keys()), 272)
            self.assertEqual(weights.get_slice("model.embed_tokens.weight").get_shape(), [49152, 576])
            self.assertEqual(weights.get_slice("model.layers.0.self_attn.q_proj.weight").get_shape(), [576, 576])
            self.assertEqual(weights.get_slice("model.layers.0.self_attn.k_proj.weight").get_shape(), [192, 576])
            self.assertNotIn("lm_head.weight", weights.keys())
            self.assertEqual(weights.get_tensor("model.norm.weight").dtype, torch.bfloat16)

    def test_saved_logits_cover_every_case(self):
        cases = json.loads((REFERENCE_DIR / "fixtures" / "cases.json").read_text())
        golden = json.loads((REFERENCE_DIR / "fixtures" / "golden.json").read_text())
        names = [case["name"] for case in cases]
        self.assertEqual(names, [case["name"] for case in golden["cases"]])
        self.assertEqual(golden["revision"], MODEL_INFO["revision"])
        with np.load(REFERENCE_DIR / "fixtures" / "logits.npz") as logits:
            self.assertEqual(set(logits.files), set(names))
            for case in golden["cases"]:
                scores = logits[case["name"]]
                self.assertEqual(scores.shape, (49152,))
                self.assertEqual(scores.dtype, np.float32)
                self.assertTrue(np.isfinite(scores).all())
                self.assertEqual(int(scores.argmax()), case["top10_next_token_ids"][0])

    def test_reference_rerun_matches_saved_results(self):
        completed = subprocess.run(
            [sys.executable, str(REFERENCE_DIR / "generate_reference.py"), "--check"],
            cwd=REFERENCE_DIR,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
