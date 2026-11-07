"""Offline checks for scripts/download_checkpoint.py (no network is touched)."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_checkpoint as dc  # noqa: E402


class ChecksumTests(unittest.TestCase):
    def test_valid_only_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.pt"
            path.write_bytes(b"checkpoint bytes")
            good = hashlib.sha256(b"checkpoint bytes").hexdigest()
            self.assertTrue(dc.is_valid(path, good))
            self.assertFalse(dc.is_valid(path, "0" * 64))

    def test_missing_file_is_not_valid(self):
        self.assertFalse(dc.is_valid(Path("does-not-exist.pt")))

    def test_download_writes_via_part_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.bin"
            source.write_bytes(b"x" * 3000)
            out = Path(tmp) / "nested" / "out.pt"
            dc.download(source.as_uri(), out)
            self.assertEqual(out.read_bytes(), b"x" * 3000)
            self.assertFalse(out.with_suffix(".pt.part").exists())


if __name__ == "__main__":
    unittest.main()
