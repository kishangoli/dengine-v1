"""Black-box checks for the standalone JSON process."""

import json
import select
import subprocess
import sys
import unittest


EXECUTABLE = sys.argv.pop(1) if len(sys.argv) > 1 else "/tmp/dengine-native"


class ProtocolTests(unittest.TestCase):
    def start(self):
        process = subprocess.Popen(
            [EXECUTABLE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self.finish, process)
        return process

    def finish(self, process):
        if process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        process.stdout.close()
        process.stderr.close()

    def response(self, process, request):
        process.stdin.write(request + "\n")
        process.stdin.flush()
        self.assertTrue(select.select([process.stdout], [], [], 3)[0], "response was not flushed")
        return json.loads(process.stdout.readline())

    def test_multiple_requests_escaping_and_recovery(self):
        process = self.start()
        prompt = '  A "quote", slash \\, Unicode café, and\nnewline  '
        first = self.response(process, json.dumps({"id": "first", "prompt": prompt}))
        self.assertEqual(first, {"id": "first", "ok": True, "text": "echo: " + prompt})

        invalid = self.response(process, '{"id":"bad","prompt":')
        self.assertEqual(invalid["error"]["code"], "invalid_json")
        self.assertIsNone(invalid["id"])

        invalid_field = self.response(process, '{"id":"field","prompt":17}')
        self.assertEqual(invalid_field["error"]["code"], "invalid_request")
        self.assertEqual(invalid_field["id"], "field")

        self.assertEqual(self.response(process, "   ")["error"]["code"], "invalid_json")
        self.assertEqual(self.response(process, "[]")["error"]["code"], "invalid_request")
        self.assertEqual(self.response(process, '{"id":"  ","prompt":"hi"}')["id"], None)

        final = self.response(process, '{"id":"last","prompt":"works"}\r')
        self.assertEqual(final, {"id": "last", "ok": True, "text": "echo: works"})
        self.assertIsNone(process.poll(), "process exited before stdin closed")

    def test_oversized_request_is_drained(self):
        process = self.start()
        prefix = '{"id":"limit","prompt":"'
        suffix = '"}'
        at_limit = prefix + "a" * (65536 - len(prefix) - len(suffix)) + suffix
        accepted = self.response(process, at_limit)
        self.assertEqual(accepted["id"], "limit")
        self.assertTrue(accepted["ok"])
        self.assertEqual(len(accepted["text"]), 6 + 65536 - len(prefix) - len(suffix))

        large = self.response(process, "x" * 65537)
        self.assertEqual(large["error"]["code"], "request_too_large")
        self.assertIsNone(large["id"])
        next_result = self.response(process, '{"id":"after","prompt":"ok"}')
        self.assertEqual(next_result, {"id": "after", "ok": True, "text": "echo: ok"})

    def test_eof_without_final_newline(self):
        process = self.start()
        output, errors = process.communicate('{"id":"eof","prompt":"last line"}')
        self.assertEqual(process.returncode, 0)
        self.assertEqual(errors, "")
        self.assertEqual(
            [json.loads(line) for line in output.splitlines()],
            [{"id": "eof", "ok": True, "text": "echo: last line"}],
        )


if __name__ == "__main__":
    unittest.main()
