"""Compare the complete C++ forward pass with saved Python model scores."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


EXECUTABLE = Path(sys.argv.pop(1)).resolve()
MODEL_DIR = Path(sys.argv.pop(1)).resolve()
FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"


def native_requests(requests):
    completed = subprocess.run(
        [str(EXECUTABLE), str(MODEL_DIR)],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode or completed.stderr:
        raise AssertionError(completed.stderr or f"exit code {completed.returncode}")
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(responses) != len(requests):
        raise AssertionError(f"expected {len(requests)} responses, got {len(responses)}")
    return responses


class ForwardReferenceTests(unittest.TestCase):
    def test_all_saved_prompts(self):
        cases = json.loads((FIXTURES / "golden.json").read_text())["cases"]
        requests = [{"input_ids": case["input_ids"], "full_scores": True}
                    for case in cases]
        responses = native_requests(requests)
        with np.load(FIXTURES / "logits.npz") as saved:
            for case, response in zip(cases, responses):
                self.assertTrue(response["ok"], (case["name"], response))
                actual = np.asarray(response["scores"], dtype=np.float32)
                expected = saved[case["name"]]
                self.assertEqual(actual.shape, expected.shape)
                maximum_error = float(np.max(np.abs(actual - expected)))
                self.assertTrue(
                    np.allclose(actual, expected, rtol=1e-4, atol=1e-3),
                    f"{case['name']}: score mismatch; maximum absolute error {maximum_error}",
                )
                self.assertEqual(response["top10_ids"], case["top10_next_token_ids"],
                                 case["name"])
                np.testing.assert_allclose(response["top10_scores"],
                                           case["top10_next_token_logits"],
                                           rtol=1e-4, atol=1e-3,
                                           err_msg=case["name"])

    def test_invalid_requests_do_not_end_process(self):
        responses = native_requests([
            {"input_ids": []},
            {"input_ids": [49152]},
            {"input_ids": [-1]},
            {"input_ids": "hello"},
            {"input_ids": [19556], "full_scores": "yes"},
            {"input_ids": [19556] * 1025},
            {"input_ids": [19556]},
        ])
        self.assertTrue(all(not response["ok"] for response in responses[:6]))
        self.assertTrue(responses[-1]["ok"])
        self.assertEqual(len(responses[-1]["top10_ids"]), 10)
        self.assertNotIn("scores", responses[-1])

    def test_missing_or_invalid_model_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            for condition in ("missing", "truncated", "incompatible"):
                if condition != "missing":
                    config = json.loads((MODEL_DIR / "config.json").read_text())
                    if condition == "incompatible":
                        config["hidden_size"] = 1
                    (model_path / "config.json").write_text(json.dumps(config))
                    (model_path / "model.safetensors").write_bytes(b"x")
                completed = subprocess.run(
                    [str(EXECUTABLE), directory], capture_output=True,
                    text=True, timeout=10,
                )
                self.assertNotEqual(completed.returncode, 0, condition)
                self.assertIn("model startup failed", completed.stderr, condition)
                self.assertEqual(completed.stdout, "", condition)


if __name__ == "__main__":
    unittest.main()
