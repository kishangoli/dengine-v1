"""Check model-backed JSON generation against pinned Python token sequences."""

import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest

from transformers import AutoTokenizer


EXECUTABLE = Path(sys.argv.pop(1)).resolve()
MODEL_DIR = Path(sys.argv.pop(1)).resolve()
FIXTURES = Path(__file__).resolve().parents[1] / "reference" / "fixtures"


def native_requests(requests):
    payload = "".join(json.dumps(request, ensure_ascii=False) + "\n"
                      if not isinstance(request, str) else request + "\n"
                      for request in requests)
    completed = subprocess.run(
        [str(EXECUTABLE), "--model", str(MODEL_DIR)], input=payload,
        capture_output=True, text=True, timeout=180,
    )
    if completed.returncode or completed.stderr:
        raise AssertionError(completed.stderr or f"exit code {completed.returncode}")
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(responses) != len(requests):
        raise AssertionError(f"expected {len(requests)} responses, got {len(responses)}")
    return responses


class GenerationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, use_fast=True,
        )

    def test_four_reference_prompts_in_one_process(self):
        cases = json.loads((FIXTURES / "cases.json").read_text())
        saved = {case["name"]: case for case in
                 json.loads((FIXTURES / "golden.json").read_text())["cases"]}
        requests = []
        for case in cases:
            if case["kind"] == "raw":
                prompt = case["text"]
            else:
                prompt = self.tokenizer.apply_chat_template(
                    case["messages"], tokenize=False, add_generation_prompt=True,
                )
            requests.append({"id": case["name"], "prompt": prompt,
                             "max_new_tokens": 8})
        responses = native_requests(requests)
        for case, response in zip(cases, responses):
            expected = saved[case["name"]]
            self.assertTrue(response["ok"], (case["name"], response))
            self.assertEqual(response["id"], case["name"])
            self.assertEqual(response["prompt_tokens"], len(expected["input_ids"]))
            self.assertEqual(response["generated_ids"], expected["generated_ids"])
            self.assertEqual(response["text"], expected["generated_text"])
            self.assertEqual(response["finish_reason"], "max_new_tokens")

    def test_limits_stop_and_recovery(self):
        prompt = "Hello, world!"
        eos_prompt = ("<|im_start|>user\nSay hello.<|im_end|>\n"
                      "<|im_start|>assistant\nThe answer is 4.")
        responses = native_requests([
            {"id": "one", "prompt": prompt, "max_new_tokens": 1},
            "{not JSON",
            {"id": "zero", "prompt": prompt, "max_new_tokens": 0},
            {"id": "too_many", "prompt": prompt, "max_new_tokens": 65},
            {"id": "bad_stop", "prompt": prompt, "stop": ""},
            {"id": "long", "prompt": "hello " * 1500},
            {"id": "stopped", "prompt": prompt, "stop": "I'm"},
            {"id": "ended", "prompt": eos_prompt, "max_new_tokens": 8},
            {"id": "again", "prompt": prompt, "max_new_tokens": 1},
        ])
        self.assertEqual(responses[0]["generated_ids"], [198])
        self.assertEqual(responses[0]["finish_reason"], "max_new_tokens")
        self.assertEqual(responses[1]["error"]["code"], "invalid_json")
        self.assertTrue(all(not response["ok"] for response in responses[2:6]))
        self.assertEqual(responses[5]["error"]["code"], "context_limit")
        self.assertTrue(responses[6]["ok"])
        self.assertEqual(responses[6]["finish_reason"], "stop_string")
        self.assertNotIn("I'm", responses[6]["text"])
        self.assertTrue(responses[7]["ok"])
        self.assertEqual(responses[7]["generated_ids"], [2])
        self.assertEqual(responses[7]["text"], "")
        self.assertEqual(responses[7]["finish_reason"], "end_token")
        self.assertEqual(responses[8]["generated_ids"], responses[0]["generated_ids"])
        self.assertEqual(responses[8]["text"], responses[0]["text"])

    def test_response_is_flushed_before_eof(self):
        process = subprocess.Popen(
            [str(EXECUTABLE), "--model", str(MODEL_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        try:
            process.stdin.write('{"id":"live","prompt":"Hello, world!","max_new_tokens":1}\n')
            process.stdin.flush()
            self.assertTrue(select.select([process.stdout], [], [], 30)[0])
            response = json.loads(process.stdout.readline())
            self.assertEqual(response["id"], "live")
            self.assertEqual(response["generated_ids"], [198])
            self.assertIsNone(process.poll())
        finally:
            process.stdin.close()
            process.wait(timeout=10)
            process.stdout.close()
            process.stderr.close()

    def test_missing_model_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [str(EXECUTABLE), "--model", directory],
                capture_output=True, text=True, timeout=10,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("generation startup failed", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
