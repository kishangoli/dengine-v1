"""Compare the C++ inspector with an independent Safetensors header/byte reader.

This integration check uses only Python's standard library. The five fixed BF16
samples were also inspected with the pinned Python safetensors package.
"""

import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = Path(sys.argv.pop(1))
MODEL_DIR = Path(sys.argv.pop(1))
WEIGHTS = MODEL_DIR / "model.safetensors"

REFERENCE_SAMPLES = {
    "model.embed_tokens.weight": ([49152, 576], 0xBDD9),
    "model.layers.0.self_attn.q_proj.weight": ([576, 576], 0xBD9A),
    "model.layers.0.self_attn.k_proj.weight": ([192, 576], 0xBDEE),
    "model.layers.17.mlp.down_proj.weight": ([576, 1536], 0xBD34),
    "model.norm.weight": ([576], 0x3FE1),
}


class ModelInspectionTests(unittest.TestCase):
    def test_all_tensors_and_reference_samples(self):
        completed = subprocess.run(
            [str(EXECUTABLE), "--inspect-model", str(MODEL_DIR)],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        report = json.loads(completed.stdout)
        self.assertEqual(report["tensor_count"], 272)
        self.assertEqual(report["parameters"], 134515008)
        self.assertEqual(report["file_bytes"], WEIGHTS.stat().st_size)
        reported = {tensor["name"]: tensor for tensor in report["tensors"]}

        with WEIGHTS.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
            data_start = 8 + header_size
            header.pop("__metadata__", None)
            self.assertEqual(set(reported), set(header))
            for name, entry in header.items():
                tensor = reported[name]
                self.assertEqual(tensor["shape"], entry["shape"], name)
                self.assertEqual(tensor["dtype"], entry["dtype"], name)
                stream.seek(data_start + entry["data_offsets"][0])
                first_bits = struct.unpack("<H", stream.read(2))[0]
                self.assertEqual(tensor["first_bf16_bits"], first_bits, name)

        for name, (shape, bits) in REFERENCE_SAMPLES.items():
            self.assertEqual(reported[name]["shape"], shape)
            self.assertEqual(reported[name]["first_bf16_bits"], bits)

    def test_corrupted_real_model_is_rejected(self):
        # Copy once; all mutations alter only the header, preserving the payload.
        with tempfile.TemporaryDirectory(prefix="dengine-stage3-") as temporary:
            directory = Path(temporary)
            shutil.copyfile(MODEL_DIR / "config.json", directory / "config.json")
            copied_weights = directory / "model.safetensors"
            shutil.copyfile(WEIGHTS, copied_weights)
            with copied_weights.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                original = json.loads(stream.read(header_size))

            def check_rejection(header, expected_error):
                encoded = json.dumps(header, separators=(",", ":")).encode()
                self.assertLessEqual(len(encoded), header_size)
                with copied_weights.open("r+b") as stream:
                    stream.seek(8)
                    stream.write(encoded.ljust(header_size, b" "))
                result = subprocess.run(
                    [str(EXECUTABLE), "--inspect-model", str(directory)],
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(expected_error, result.stderr)

            wrong_type = json.loads(json.dumps(original))
            wrong_type["model.norm.weight"]["dtype"] = "F16"
            check_rejection(wrong_type, "dtype")

            wrong_shape = json.loads(json.dumps(original))
            wrong_shape["model.embed_tokens.weight"]["shape"][0] += 1
            check_rejection(wrong_shape, "shape")

            missing = json.loads(json.dumps(original))
            missing.pop("model.norm.weight")
            check_rejection(missing, "tensor count")

            overlap = json.loads(json.dumps(original))
            names_by_offset = sorted(
                (name for name in overlap if name != "__metadata__"),
                key=lambda name: overlap[name]["data_offsets"][0],
            )
            second = overlap[names_by_offset[1]]["data_offsets"]
            second[0] -= 2
            second[1] -= 2
            check_rejection(overlap, "gap or overlap")

            with copied_weights.open("r+b") as stream:
                stream.seek(8)
                encoded = json.dumps(original, separators=(",", ":")).encode()
                stream.write(encoded.ljust(header_size, b" "))
                stream.truncate(WEIGHTS.stat().st_size - 1)
            result = subprocess.run(
                [str(EXECUTABLE), "--inspect-model", str(directory)],
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("range", result.stderr)


if __name__ == "__main__":
    unittest.main()
