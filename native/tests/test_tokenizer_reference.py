"""Compare native tokenization with the pinned Python Transformers tokenizer."""

import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from transformers import AutoTokenizer


EXECUTABLE = Path(sys.argv.pop(1)).resolve()
MODEL_DIR = Path(sys.argv.pop(1)).resolve()
GOLDEN = Path(__file__).resolve().parents[1] / "reference" / "fixtures" / "golden.json"
CASES = GOLDEN.with_name("cases.json")


def native_requests(requests):
    completed = subprocess.run(
        [str(EXECUTABLE), str(MODEL_DIR)],
        input="".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr)
    if completed.stderr:
        raise AssertionError(completed.stderr)
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(responses) != len(requests):
        raise AssertionError(f"expected {len(requests)} responses; got {len(responses)}")
    return responses


class TokenizerReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, use_fast=True
        )

    def test_raw_text_and_saved_golden_inputs(self):
        texts = [
            "",
            "Hello, world!",
            "  The quick brown fox\n",
            "Café ☕ — привет",
            "hello",
            "hello world",
            " we're I'LL can't it's they\'ve",
            "123 456 7890",
            "a1b 2.5 -42",
            "   ",
            "\n\n\t \r\n",
            "word\r\nother",
            "punctuation!?...—:;",
            "中文 日本語 한글 Ελληνικά हिन्दी العربية",
            "👨‍👩‍👧‍👦 😀 🧑🏽‍💻",
            "aⅣⅤb a²³b a٣٤b",
            "e\u0301 and é",
            "\u00a0\u200b\u2028\u2029",
            "\x00\x04\x06control",
            "!\x04!",
            "<|im_start|>user\nHi<|im_end|>",
            "<repo_name>project<file_sep>main.py",
            "not-a-token <|unknown|>",
        ]
        golden = json.loads(GOLDEN.read_text())
        fixture_cases = json.loads(CASES.read_text())
        saved_by_name = {case["name"]: case for case in golden["cases"]}
        saved_raw = {case["text"]: saved_by_name[case["name"]]
                     for case in fixture_cases if case["kind"] == "raw"}
        for case in fixture_cases:
            if case["kind"] == "raw" and case["text"] not in texts:
                texts.append(case["text"])
        # Adjacent number, word, punctuation, and whitespace boundaries expose
        # pre-tokenizer differences that a few hand-picked sentences can miss.
        for left in ("a", " ", "\n", "é", "界", "👋"):
            for right in ("123", "!", "word", " \n"):
                texts.append(left + right)
        generator = random.Random(44)
        alphabet = ["a", "Z", "0", "9", " ", "\n", "\r", "\t", "!", ".", "'",
                    "é", "e\u0301", "☕", "👋", "👨", "\u200d", "中", "あ", "Ω",
                    "Ж", "क", "ع", "١", "Ⅳ", "²", "\u00a0", "\u200b",
                    "\u2028", "\u2029", "\x04", "\x06"]
        texts.extend("".join(generator.choice(alphabet)
                             for _ in range(generator.randrange(0, 35)))
                     for _ in range(100))

        responses = native_requests([{"op": "encode", "text": text} for text in texts])
        for text, response in zip(texts, responses):
            self.assertTrue(response["ok"], (text, response))
            expected = self.reference(text, add_special_tokens=False).input_ids
            self.assertEqual(response["ids"], expected, repr(text))
            if text in saved_raw:
                self.assertEqual(response["ids"], saved_raw[text]["input_ids"])

    def test_decoding_and_special_tokens(self):
        texts = [
            "Hello, world!", "Café ☕ — привет", "👨‍👩‍👧‍👦", "  spaces\n",
            "123 <|im_start|>user\nHi<|im_end|>", "e\u0301", "\x04",
        ]
        sequences = [self.reference(text, add_special_tokens=False).input_ids for text in texts]
        golden = json.loads(GOLDEN.read_text())
        sequences += [case["generated_ids"] for case in golden["cases"]]
        sequences += [[1, 2], [0, 1, 4093, 198, 2], [49151], [], [173]]
        generator = random.Random(77)
        sequences.extend([generator.randrange(49152)
                          for _ in range(generator.randrange(0, 15))]
                         for _ in range(50))
        requests = [
            {"op": "decode", "ids": ids, "skip_special_tokens": skip}
            for ids in sequences for skip in (False, True)
        ]
        responses = native_requests(requests)
        for request, response in zip(requests, responses):
            self.assertTrue(response["ok"], response)
            expected = self.reference.decode(
                request["ids"],
                skip_special_tokens=request["skip_special_tokens"],
                clean_up_tokenization_spaces=False,
            )
            self.assertEqual(response["text"], expected, request)

    def test_chat_formatting_and_ids(self):
        chats = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "system", "content": "Be concise."},
             {"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "Hello!"},
             {"role": "user", "content": "And now?"}],
            [{"role": "user", "content": "Unicode ☕ and\nnewline"}],
        ]
        fixture = next(case for case in json.loads(CASES.read_text()) if case["kind"] == "chat")
        chats.append(fixture["messages"])
        requests = [
            {"op": "chat", "messages": messages, "add_generation_prompt": prompt}
            for messages in chats for prompt in (False, True)
        ]
        responses = native_requests(requests)
        for request, response in zip(requests, responses):
            self.assertTrue(response["ok"], response)
            formatted = self.reference.apply_chat_template(
                request["messages"], tokenize=False,
                add_generation_prompt=request["add_generation_prompt"],
            )
            self.assertEqual(response["formatted"], formatted)
            expected_ids = self.reference.apply_chat_template(
                request["messages"], tokenize=True,
                add_generation_prompt=request["add_generation_prompt"],
            )
            self.assertEqual(response["ids"], expected_ids)
        golden_chat = next(case for case in json.loads(GOLDEN.read_text())["cases"]
                           if case["name"] == "chat")
        self.assertEqual(responses[-1]["ids"], golden_chat["input_ids"])

        empty = native_requests([
            {"op": "chat", "messages": [], "add_generation_prompt": False},
            {"op": "chat", "messages": [], "add_generation_prompt": True},
        ])
        self.assertEqual(empty[0]["formatted"], "")
        self.assertEqual(empty[0]["ids"], [])
        self.assertEqual(empty[1]["formatted"], "<|im_start|>assistant\n")
        self.assertEqual(empty[1]["ids"],
                         self.reference("<|im_start|>assistant\n",
                                        add_special_tokens=False).input_ids)

    def test_invalid_requests_are_contained(self):
        responses = native_requests([
            {"op": "decode", "ids": [49152]},
            {"op": "decode", "ids": [-1]},
            {"op": "chat", "messages": [{"role": 7, "content": "bad"}]},
            {"op": "unknown"},
            {"op": "encode", "text": "after errors"},
        ])
        self.assertTrue(all(not response["ok"] for response in responses[:4]))
        self.assertTrue(responses[-1]["ok"])
        self.assertEqual(responses[-1]["ids"],
                         self.reference("after errors", add_special_tokens=False).input_ids)

    def test_incompatible_tokenizer_files_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dengine-tokenizer-test-") as temporary:
            directory = Path(temporary)
            config_path = directory / "tokenizer_config.json"
            data_path = directory / "tokenizer.json"
            shutil.copyfile(MODEL_DIR / "tokenizer_config.json", config_path)
            shutil.copyfile(MODEL_DIR / "tokenizer.json", data_path)

            def fails_to_start():
                completed = subprocess.run(
                    [str(EXECUTABLE), str(directory)], input="", text=True,
                    capture_output=True, timeout=15,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("tokenizer startup failed", completed.stderr)

            config = json.loads(config_path.read_text())
            config["chat_template"] = "different template"
            config_path.write_text(json.dumps(config))
            fails_to_start()
            shutil.copyfile(MODEL_DIR / "tokenizer_config.json", config_path)

            data = json.loads(data_path.read_text())
            data["model"]["merges"].pop()
            data_path.write_text(json.dumps(data))
            fails_to_start()
            data_path.unlink()
            fails_to_start()


if __name__ == "__main__":
    unittest.main()
