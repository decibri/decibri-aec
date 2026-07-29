#!/usr/bin/env python3
"""Automated refusal tests for benchmarks/run_aecmos.py.

Each test builds a throwaway internal result and triplet fixtures in a
temporary directory, runs the scorer against it as a subprocess, and asserts
three things: the scorer refuses, it exits non-zero, and it writes no scored
result. Nothing under data/ is read or written.

Run from the crate root:

    benchmarks\\.venv\\Scripts\\python.exe benchmarks\\test_scorer_refusals.py
    benchmarks\\.venv\\Scripts\\python.exe -m unittest discover -s benchmarks
"""

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parent.parent
SCORER = CRATE_ROOT / "benchmarks" / "run_aecmos.py"
MODEL = CRATE_ROOT / "models" / "Run_1663915512_Stage_0.onnx"
MODEL_RATE = 16000

# The scorer needs these three packages before it reaches any refusal path.
REQUIRED = ("numpy", "librosa", "onnxruntime")


def missing_requirements():
    absent = []
    for name in REQUIRED:
        try:
            __import__(name)
        except ImportError:
            absent.append(name)
    return absent


def write_wav(path, samples, rate):
    """Writes mono 16-bit PCM."""
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        )


def tone(n, rate, freq=440.0):
    return [0.4 * math.sin(2.0 * math.pi * freq * i / rate) for i in range(n)]


def aligned_clip(dirpath, cid, scenario, talk_type, lengths, rates, declared):
    """One protocol 2 clip record plus its triplet on disk. `lengths` and
    `rates` give the per-key sample count and rate; `declared` gives the
    `sample_rate_hz` and `samples` the record claims."""
    rec = {"id": cid, "scenario": scenario, "talk_type": talk_type, "aligned": {}}
    for key in ("mic", "lpb", "enhanced"):
        path = dirpath / f"{cid}_{key}.wav"
        write_wav(path, tone(lengths[key], rates[key]), rates[key])
        rec["aligned"][key] = str(path)
    rec["aligned"]["sample_rate_hz"] = declared["sample_rate_hz"]
    if declared.get("samples") is not None:
        rec["aligned"]["samples"] = declared["samples"]
    return rec


def legacy_clip(dirpath, cid, scenario, talk_type, n=8000, rate=MODEL_RATE):
    """One protocol 1 clip record (no aligned triplet) plus its files."""
    rec = {"id": cid, "scenario": scenario, "talk_type": talk_type}
    for key in ("mic", "lpb", "enhanced"):
        path = dirpath / f"{cid}_legacy_{key}.wav"
        write_wav(path, tone(n, rate), rate)
        rec[key] = str(path)
    return rec


def write_result(dirpath, clips, set_name="refusal-fixture"):
    path = dirpath / "bench-00000000T000000Z-refusal-fixture.json"
    payload = {
        "schema": "decibri-aec-bench/internal/v3",
        "protocol": 2,
        "input": {"set_name": set_name},
        "source": {"combined_sha256": "0" * 64},
        "clips": clips,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@unittest.skipUnless(MODEL.is_file(), f"AECMOS model not present at {MODEL}")
@unittest.skipIf(missing_requirements(), f"missing packages: {missing_requirements()}")
class ScorerRefusalTests(unittest.TestCase):
    def run_scorer(self, results_path, extra=()):
        """Runs the scorer over one internal result and returns
        (exit_code, combined_output, new_files_written)."""
        before = {p.name for p in results_path.parent.iterdir()}
        proc = subprocess.run(
            [sys.executable, str(SCORER), "--results", str(results_path), "--workers", "1"]
            + list(extra),
            cwd=str(CRATE_ROOT),
            capture_output=True,
            text=True,
        )
        after = {p.name for p in results_path.parent.iterdir()}
        return proc.returncode, proc.stdout + proc.stderr, after - before

    def assert_refused(self, code, output, new_files, fragment):
        self.assertNotEqual(code, 0, f"scorer exited 0; output:\n{output}")
        self.assertIn("REFUSED", output, f"no refusal in output:\n{output}")
        self.assertIn(fragment, output, f"expected '{fragment}' in output:\n{output}")
        # A resume checkpoint may be written; a scored result must not be.
        scored = [n for n in new_files if not n.startswith("aecmos-partial-")]
        self.assertEqual(scored, [], f"scorer wrote a result: {scored}")
        self.assertNotIn("results:", output, f"scorer reported a result:\n{output}")

    def test_wellformed_record_scores_and_writes(self):
        """The control for the four refusal cases: a consistent protocol 2
        record scores, exits zero, and writes a result."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            clip = aligned_clip(
                d,
                "wellformed",
                "farend-singletalk",
                "st",
                dict.fromkeys(("mic", "lpb", "enhanced"), 4 * MODEL_RATE),
                dict.fromkeys(("mic", "lpb", "enhanced"), MODEL_RATE),
                {"sample_rate_hz": MODEL_RATE, "samples": 4 * MODEL_RATE},
            )
            code, out, new = self.run_scorer(write_result(d, [clip]))
            self.assertEqual(code, 0, f"scorer did not succeed; output:\n{out}")
            self.assertNotIn("REFUSED", out)
            written = sorted(n for n in new if n.startswith("aecmos-"))
            self.assertEqual(
                len(written), 2, f"expected a JSON and a text result, got {written}"
            )

    def test_mismatched_triplet_length_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # The enhanced signal is shorter than the record declares.
            clip = aligned_clip(
                d,
                "len-vs-declared",
                "farend-singletalk",
                "st",
                {"mic": 8000, "lpb": 8000, "enhanced": 7000},
                dict.fromkeys(("mic", "lpb", "enhanced"), MODEL_RATE),
                {"sample_rate_hz": MODEL_RATE, "samples": 8000},
            )
            code, out, new = self.run_scorer(write_result(d, [clip]))
            self.assert_refused(code, out, new, "length 7000 differs from the declared 8000")

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # No declared count, and the three signals disagree with each other.
            clip = aligned_clip(
                d,
                "len-vs-each-other",
                "farend-singletalk",
                "st",
                {"mic": 8000, "lpb": 8000, "enhanced": 7000},
                dict.fromkeys(("mic", "lpb", "enhanced"), MODEL_RATE),
                {"sample_rate_hz": MODEL_RATE, "samples": None},
            )
            code, out, new = self.run_scorer(write_result(d, [clip]))
            self.assert_refused(code, out, new, "triplet lengths differ")

    def test_mismatched_sample_rate_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # One triplet member is at a different rate from the declared one.
            clip = aligned_clip(
                d,
                "rate-vs-declared",
                "farend-singletalk",
                "st",
                dict.fromkeys(("mic", "lpb", "enhanced"), 8000),
                {"mic": MODEL_RATE, "lpb": MODEL_RATE, "enhanced": 8000},
                {"sample_rate_hz": MODEL_RATE, "samples": 8000},
            )
            code, out, new = self.run_scorer(write_result(d, [clip]))
            self.assert_refused(code, out, new, "sample rate 8000 differs from the declared")

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # The declared triplet rate is not the model's rate.
            clip = aligned_clip(
                d,
                "rate-vs-model",
                "farend-singletalk",
                "st",
                dict.fromkeys(("mic", "lpb", "enhanced"), 8000),
                dict.fromkeys(("mic", "lpb", "enhanced"), 8000),
                {"sample_rate_hz": 8000, "samples": 8000},
            )
            code, out, new = self.run_scorer(write_result(d, [clip]))
            self.assert_refused(code, out, new, "does not match the model rate")

    def test_raw_condition_against_protocol_1_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            clips = [legacy_clip(d, "legacy-only", "farend-singletalk", "st")]
            code, out, new = self.run_scorer(
                write_result(d, clips), extra=("--condition", "raw")
            )
            self.assert_refused(code, out, new, "the raw condition needs a protocol 2 result")

    def test_mixed_aligned_and_legacy_record_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            aligned = aligned_clip(
                d,
                "aligned-entry",
                "farend-singletalk",
                "st",
                dict.fromkeys(("mic", "lpb", "enhanced"), 8000),
                dict.fromkeys(("mic", "lpb", "enhanced"), MODEL_RATE),
                {"sample_rate_hz": MODEL_RATE, "samples": 8000},
            )
            legacy = legacy_clip(d, "legacy-entry", "doubletalk", "dt")
            code, out, new = self.run_scorer(write_result(d, [aligned, legacy]))
            self.assert_refused(code, out, new, "mixes aligned (protocol 2) and legacy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
