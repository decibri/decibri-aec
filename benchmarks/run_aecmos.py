#!/usr/bin/env python3
"""Optional AECMOS scoring step for the decibri-aec benchmark kit.

Scores the enhanced WAVs of one internal benchmark run with Microsoft's
AECMOS ONNX model, the objective metric used to rank submissions in the
Microsoft AEC Challenges. It reads the newest internal result file (or the
one named with --results), loads each clip's original loopback and
microphone recordings plus the kit's enhanced output, trims to the rated
region the AECMOS README requires for the ICASSP 2022 test set, and emits
EchoMOS and DegMOS per clip and per scenario, alongside a preserved JSON
result stamped with the model hash and the scored portions.

The feature pipeline mirrors the MIT-licensed reference implementation in
the AEC-Challenge repository (AECMOS/AECMOS_local/aecmos.py): audio loaded
at the model's own sample rate, a 160-band log-mel spectrogram, the
scenario marker rows for scenario-aware models, and a 20 second cap. The
torch dependency of the reference script is not needed; the GRU state is a
plain zeros array.

Usage, from the crate root:

    python benchmarks/run_aecmos.py [--results PATH] [--model PATH]
    python benchmarks/run_aecmos.py --split MANIFEST.json [--run LABEL]

The first form scores a standalone internal result (the newest, or the one
named with --results). The second form scores a split-manifest run: it reads
the internal result the benchmark recorded for that run, scores it, and fills
the SAME manifest result entry's aecmos field with per-scenario EchoMOS and
DegMOS, so one entry ends up holding both the internal and the AECMOS numbers.
The fill is a targeted textual replacement of that entry's `"aecmos": null`, so
every frozen split field stays byte-identical.

Artifacts are written beside the internal result they scored. When that result
is a run folder's bench.json they take the fixed names aecmos.json and
aecmos-raw.json, so one run folder holds one file per step; otherwise the run
stamp and set name go in the file name. A manifest's internal_result_file is
accepted either as a bare name under data/bench-output/results or as a
crate-root-relative path, so records from both layouts stay readable.

Protocol 2 internal results carry a canonical aligned triplet per clip
(mic, loopback, enhanced, all at the engine rate on one shared timeline).
For those records this script loads the triplet directly and performs NO
resampling of its own, and it REFUSES a record, aborting the run, when the
triplet's lengths differ, its sample rates differ, or the declared alignment
metadata is inconsistent. Legacy records without a triplet keep the original
load-and-resample path and are marked protocol 1 in the output.

--condition selects what is scored against the triplet's references:
`decibri` (default) scores the enhanced signal, `raw` scores the canonical
microphone itself, so the two conditions differ only by the AEC. The raw
condition requires protocol 2 records and never touches a split manifest.

The model is read from models/Run_1663915512_Stage_0.onnx by default and
is never downloaded by this script; see benchmarks/ATTRIBUTIONS.md for
where it comes from. If the model file is absent this script exits cleanly
and the internal benchmark results stand on their own.

Dependencies (only for this optional step, never for the Rust crate):

    pip install numpy librosa onnxruntime
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = CRATE_ROOT / "models" / "Run_1663915512_Stage_0.onnx"
DEFAULT_RESULTS_DIR = CRATE_ROOT / "data" / "bench-output" / "results"

# The benchmark's --run-dir layout puts one run's whole output in one folder and
# names the internal result exactly this. Artifacts written beside such a file
# take fixed names too, so a run folder holds one file per step.
RUN_DIR_BENCH_NAME = "bench.json"

# Model parameters per the reference implementation's table. The file name
# selects the parameter set, exactly as the reference script matches on the
# model path.
MODEL_PARAMS = {
    "Run_1663915512_Stage_0.onnx": {
        "sr": 16000,
        "dft_size": 512,
        "scenario_marker": True,
        "hidden": (4, 1, 64),
        "note": "AECMOS v4, 16 kHz, scenario-aware",
    },
    "Run_1663829550_Stage_0.onnx": {
        "sr": 16000,
        "dft_size": 512,
        "scenario_marker": False,
        "hidden": (4, 1, 64),
        "note": "AECMOS v4 no_scenarios, 16 kHz",
    },
    "Run_1668423760_Stage_0.onnx": {
        "sr": 48000,
        "dft_size": 1536,
        "scenario_marker": True,
        "hidden": (4, 1, 96),
        "note": "AECMOS 48 kHz, scenario-aware",
    },
}

# Seconds the reference implementation caps every scored segment at.
MAX_SCORED_SECONDS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a decibri-aec benchmark run with AECMOS."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Internal result JSON to score (default: newest bench-*.json "
        "under data/bench-output/results).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the AECMOS ONNX model "
        "(default: models/Run_1663915512_Stage_0.onnx).",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Split manifest to score. Fills the target run entry's aecmos "
        "field with per-scenario EchoMOS and DegMOS.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="With --split, the run label to score (default: the most recent "
        "entry whose aecmos is still null).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the clips processed (0 means all): a stratified total drawn "
        "across the scenarios in the set's proportions, id-sorted within each, "
        "so it is deterministic and matches the benchmark's own --limit.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=("decibri", "raw"),
        default="decibri",
        help="Normal scoring only: score the enhanced signal (decibri, the "
        "default) or the canonical microphone itself (raw). raw requires "
        "protocol 2 aligned records and never touches a split manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Normal scoring only: score across this many spawned worker "
        "processes (1 = single process).",
    )
    parser.add_argument(
        "--worker-threads",
        dest="worker_threads",
        type=int,
        default=1,
        help="Normal scoring only: onnxruntime intra/inter op thread cap inside "
        "each worker (0 = onnxruntime default).",
    )
    return parser.parse_args()


def resolve_internal_ref(ref):
    """A manifest's `internal_result_file` is either a bare file name written by
    the --out-root layout, or a crate-root-relative path written by the
    --run-dir layout. Both forms resolve here, so records from either layout
    stay readable."""
    candidates = [DEFAULT_RESULTS_DIR / ref, CRATE_ROOT / ref]
    for path in candidates:
        if path.is_file():
            return path, None
    tried = "\n".join(f"    {p}" for p in candidates)
    return None, f"internal result '{ref}' not found. Tried:\n{tried}"


def in_run_dir(results_path):
    """True when this internal result is a run folder's `bench.json`."""
    return results_path.name == RUN_DIR_BENCH_NAME


def guard_out_dir(out_dir):
    """Nothing this script writes belongs in the crate root."""
    try:
        resolved = out_dir.resolve()
    except OSError:
        resolved = out_dir
    if resolved == CRATE_ROOT:
        print("AECMOS: refusing to write to the crate root; score a result "
              "that lives under data/.")
        raise SystemExit(2)


def newest_results_file():
    if not DEFAULT_RESULTS_DIR.is_dir():
        return None
    candidates = sorted(DEFAULT_RESULTS_DIR.glob("bench-*.json"))
    return candidates[-1] if candidates else None


def load_manifest(path):
    """Reads a split manifest for inspection (never for rewriting)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def select_entry(manifest, run_label):
    """Picks the result entry to score. With a run label, the last entry that
    names it; otherwise the last entry whose aecmos is still null. Returns
    (entry, None) or (None, message)."""
    results = manifest.get("results", [])
    if not results:
        return None, (
            "manifest has no results entries; run the benchmark in --split "
            "mode first to record one"
        )
    candidates = results
    if run_label is not None:
        candidates = [e for e in results if e.get("run") == run_label]
        if not candidates:
            return None, f"no result entry with run '{run_label}' in the manifest"
    pending = [e for e in candidates if e.get("aecmos") is None]
    if pending:
        return pending[-1], None
    return candidates[-1], None


def splice_aecmos(manifest_path, entry, aecmos_obj):
    """Replaces the target entry's `"aecmos": null` with `aecmos_obj`, touching
    nothing else. Reads and writes with newline translation disabled so every
    other byte, including all frozen split fields, is preserved exactly.
    Returns (True, None) on success or (False, message)."""
    with open(manifest_path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    lines = text.split("\n")
    target_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip().rstrip(",")
        if not (stripped.startswith("{") and '"run"' in stripped and '"date"' in stripped):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if (
            obj.get("run") == entry.get("run")
            and obj.get("date") == entry.get("date")
            and obj.get("set") == entry.get("set")
        ):
            target_idx = idx
            break
    if target_idx is None:
        return False, "could not locate the target entry in the manifest text"
    needle = '"aecmos": null'
    if needle not in lines[target_idx]:
        return False, "the target entry already holds an aecmos value; not overwriting"
    replacement = '"aecmos": ' + json.dumps(aecmos_obj)
    lines[target_idx] = lines[target_idx].replace(needle, replacement, 1)
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines))
    return True, None


def rated_region(talk_type, n_samples, sr):
    """Start sample of the rated region for `talk_type`, per the AECMOS README
    trimming rule for the ICASSP 2022 test set. Returns (start_index, note)."""
    seconds = n_samples / sr
    if talk_type == "st":
        return n_samples // 2, None
    if talk_type == "dt":
        rated = (seconds - 15.0) / 2.0
        if rated <= 0:
            return 0, (
                "clip is shorter than the doubletalk trimming rule assumes; "
                "the whole clip was scored"
            )
        return n_samples - int(rated * sr), None
    return 0, None


def mel_features(signal, sr, dft_size, np, librosa):
    """Log-mel feature matrix for one signal, in the reference layout."""
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=dft_size + 1,
        hop_length=int(0.5 * dft_size),
        n_mels=160,
    )
    mel = (librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0
    return mel.T


def score(talk_type, lpb, mic, enh, session, params, sr, dft_size, np, librosa):
    """EchoMOS and DegMOS for one (loopback, microphone, enhanced) triple. Both
    conditions go through this, so their numbers are produced by identical
    feature and model math."""
    lpb_f = mel_features(lpb, sr, dft_size, np, librosa)
    mic_f = mel_features(mic, sr, dft_size, np, librosa)
    enh_f = mel_features(enh, sr, dft_size, np, librosa)
    if params["scenario_marker"]:
        ne_st, fe_st = {"nst": (1, 0), "st": (0, 1), "dt": (0, 0)}[talk_type]
        width = mic_f.shape[1]
        mic_f = np.concatenate(
            (mic_f, np.ones((20, width)) * (1 - fe_st), np.zeros((20, width))), axis=0
        )
        lpb_f = np.concatenate(
            (lpb_f, np.ones((20, width)) * (1 - ne_st), np.zeros((20, width))), axis=0
        )
        enh_f = np.concatenate(
            (enh_f, np.ones((20, width)), np.zeros((20, width))), axis=0
        )
    feats = np.expand_dims(np.stack((lpb_f, mic_f, enh_f)), axis=0).astype(np.float32)
    h0 = np.zeros(params["hidden"], dtype=np.float32)
    feed = {}
    for inp in session.get_inputs():
        feed[inp.name] = h0 if inp.name == "h0" else feats
    raw = np.ravel(session.run(None, feed)[0])
    return float(raw[0]), float(raw[1])


# How many completed clips between resume-checkpoint writes.
CHECKPOINT_EVERY = 25

# Marker prefix for a hard triplet refusal; the run aborts when one is raised.
TRIPLET_PREFIX = "TRIPLET-REFUSED: "


def refuse(message):
    """Raises the hard refusal that aborts the run."""
    raise ValueError(TRIPLET_PREFIX + message)


def resolve_path(rel):
    """A path from the internal result, resolved against the crate root."""
    p = Path(rel)
    return p if p.is_absolute() else CRATE_ROOT / p


def load_triplet(clip, aligned, condition, librosa, model_sr):
    """Loads one canonical aligned triplet with NO resampling and returns
    (lpb, mic, enh) for the chosen condition. Refuses the record, aborting
    the run, when lengths, rates, or the declared metadata disagree."""
    cid = clip.get("id", "?")
    declared_sr = aligned.get("sample_rate_hz")
    declared_n = aligned.get("samples")
    if declared_sr != model_sr:
        refuse(
            f"{cid}: declared triplet rate {declared_sr} does not match the "
            f"model rate {model_sr}"
        )
    loaded = {}
    for key in ("mic", "lpb", "enhanced"):
        rel = aligned.get(key)
        if not rel:
            refuse(f"{cid}: aligned record is missing '{key}'")
        p = resolve_path(rel)
        if not p.is_file():
            refuse(f"{cid}: aligned file missing: {p}")
        y, file_sr = librosa.load(str(p), sr=None)
        if file_sr != declared_sr:
            refuse(
                f"{cid}: {key} sample rate {file_sr} differs from the "
                f"declared {declared_sr}"
            )
        if declared_n is not None and len(y) != declared_n:
            refuse(
                f"{cid}: {key} length {len(y)} differs from the declared "
                f"{declared_n}"
            )
        loaded[key] = y
    n_mic, n_lpb, n_enh = len(loaded["mic"]), len(loaded["lpb"]), len(loaded["enhanced"])
    if not (n_mic == n_lpb == n_enh):
        refuse(
            f"{cid}: triplet lengths differ (mic {n_mic}, lpb {n_lpb}, "
            f"enhanced {n_enh})"
        )
    enh = loaded["mic"] if condition == "raw" else loaded["enhanced"]
    return loaded["lpb"], loaded["mic"], enh


def allocate(sizes, target):
    """Splits `target` across buckets in proportion to their sizes with
    largest-remainder rounding, so the parts sum to min(target, total) and no
    bucket exceeds its size. Mirrors the allocation the benchmark and make-split
    use, so a limit selects the same clips here as there."""
    total = sum(sizes)
    if total == 0 or target <= 0:
        return [0] * len(sizes)
    target = min(target, total)
    alloc = []
    remainders = []
    assigned = 0
    for i, size in enumerate(sizes):
        quota = target * size / total
        floor = min(int(quota), size)
        alloc.append(floor)
        assigned += floor
        remainders.append((quota - int(quota), i))
    remainders.sort(key=lambda t: (-t[0], t[1]))
    leftover = target - assigned
    progressed = True
    while leftover > 0 and progressed:
        progressed = False
        for _, i in remainders:
            if leftover == 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                leftover -= 1
                progressed = True
    return alloc


def stratified_clip_limit(clips, limit):
    """Reduces `clips` to `limit` total, stratified across scenarios by
    `allocate` and taking the id-sorted prefix within each, in the benchmark's
    scenario order. Original clip order is preserved in the returned list."""
    order = ["farend-singletalk", "doubletalk", "nearend-singletalk"]
    groups = {
        s: sorted((c for c in clips if c.get("scenario") == s), key=lambda c: c["id"])
        for s in order
    }
    sizes = [len(groups[s]) for s in order]
    keep = allocate(sizes, limit)
    chosen = set()
    for s, k in zip(order, keep):
        for c in groups[s][:k]:
            chosen.add(c["id"])
    return [c for c in clips if c["id"] in chosen]


def checkpoint_path(results_path, condition):
    """The resume checkpoint for one internal result and condition,
    alongside it."""
    suffix = "" if condition == "decibri" else f"-{condition}"
    return results_path.parent / f"aecmos-partial-{results_path.stem}{suffix}.json"


def load_checkpoint(path, internal_name, model_sha, condition):
    """Completed per-clip entries from a checkpoint, keyed by id, but only when
    it belongs to this internal result, model, and condition; otherwise
    empty."""
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    if d.get("internal_result") != internal_name or d.get("model_sha256") != model_sha:
        return {}
    if d.get("condition", "decibri") != condition:
        return {}
    return {e["id"]: e for e in d.get("scored", [])}


def save_checkpoint(path, internal_name, model_sha, condition, done):
    """Writes the checkpoint via a temp file and atomic replace."""
    payload = {
        "schema": "decibri-aec-bench/aecmos-partial/v1",
        "internal_result": internal_name,
        "model_sha256": model_sha,
        "condition": condition,
        "scored": list(done.values()),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


# Per-worker state, populated once by worker_init and read by worker_score. With
# the spawn start method each worker builds its own session and library handles.
_WK = {}


def worker_init(model_path_str, worker_threads, condition="decibri"):
    import numpy as np
    import librosa
    import onnxruntime as ort

    params = MODEL_PARAMS[Path(model_path_str).name]
    _WK["np"] = np
    _WK["librosa"] = librosa
    _WK["params"] = params
    _WK["sr"] = params["sr"]
    _WK["dft"] = params["dft_size"]
    _WK["condition"] = condition
    if worker_threads and worker_threads > 0:
        so = ort.SessionOptions()
        so.intra_op_num_threads = worker_threads
        so.inter_op_num_threads = worker_threads
        _WK["session"] = ort.InferenceSession(model_path_str, sess_options=so)
    else:
        _WK["session"] = ort.InferenceSession(model_path_str)


def worker_score(clip):
    """Scores one clip. Returns (entry_or_None, note_or_None); the entry matches
    the per-clip record the single-process path produced. A protocol 2 record
    is loaded from its canonical aligned triplet with no resampling; a legacy
    record keeps the original load-and-resample path."""
    np = _WK["np"]
    librosa = _WK["librosa"]
    session = _WK["session"]
    params = _WK["params"]
    sr = _WK["sr"]
    dft = _WK["dft"]
    condition = _WK.get("condition", "decibri")
    talk_type = clip["talk_type"]
    aligned = clip.get("aligned")
    if aligned is not None:
        lpb, mic, enh = load_triplet(clip, aligned, condition, librosa, sr)
        n = len(mic)
    else:
        if condition == "raw":
            refuse(
                f"{clip.get('id', '?')}: the raw condition needs a protocol 2 "
                "aligned record; this record has none"
            )
        lpb_p = resolve_path(clip["lpb"])
        mic_p = resolve_path(clip["mic"])
        enh_p = resolve_path(clip["enhanced"])
        for p in (lpb_p, mic_p, enh_p):
            if not p.is_file():
                return None, f"{clip['id']}: missing {p}"
        lpb, _ = librosa.load(str(lpb_p), sr=sr)
        mic, _ = librosa.load(str(mic_p), sr=sr)
        enh, _ = librosa.load(str(enh_p), sr=sr)
        n = min(len(lpb), len(mic), len(enh))
        lpb, mic, enh = lpb[:n], mic[:n], enh[:n]
    start, note = rated_region(talk_type, n, sr)
    end = min(n, start + MAX_SCORED_SECONDS * sr)
    echo_mos, deg_mos = score(
        talk_type, lpb[start:end], mic[start:end], enh[start:end],
        session, params, sr, dft, np, librosa,
    )
    entry = {
        "id": clip["id"],
        "scenario": clip["scenario"],
        "talk_type": talk_type,
        "echo_mos": round(echo_mos, 4),
        "deg_mos": round(deg_mos, 4),
        "scored_start_s": round(start / sr, 3),
        "scored_end_s": round(end / sr, 3),
    }
    return entry, (f"{clip['id']}: {note}" if note else None)


def main():
    args = parse_args()

    model_path = args.model
    if not model_path.is_absolute():
        model_path = CRATE_ROOT / model_path
    if not model_path.is_file():
        print("AECMOS: skipped, model not found.")
        print(f"  expected at: {model_path}")
        print(
            "  Place Microsoft's AECMOS model there (see "
            "benchmarks/ATTRIBUTIONS.md for its origin); the internal "
            "benchmark results are complete without this step."
        )
        return 0

    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    try:
        import onnxruntime as ort
    except ImportError:
        missing.append("onnxruntime")
    if missing:
        print("AECMOS: cannot run, missing Python packages: " + ", ".join(missing))
        print("  install with: pip install " + " ".join(missing))
        print("  The internal benchmark results are complete without this step.")
        return 2

    params = MODEL_PARAMS.get(model_path.name)
    if params is None:
        print(f"AECMOS: unrecognized model file name '{model_path.name}'.")
        print("  Supported: " + ", ".join(sorted(MODEL_PARAMS)))
        return 2
    sr = params["sr"]
    dft_size = params["dft_size"]

    # Split mode resolves the internal result from the manifest entry to score;
    # standalone mode resolves it from --results or the newest file.
    manifest_path = None
    target_entry = None
    if args.split is not None:
        manifest_path = args.split
        if not manifest_path.is_absolute():
            manifest_path = CRATE_ROOT / manifest_path
        if not manifest_path.is_file():
            print(f"AECMOS: split manifest not found: {manifest_path}")
            return 2
        manifest = load_manifest(manifest_path)
        target_entry, err = select_entry(manifest, args.run)
        if err:
            print(f"AECMOS: {err}")
            return 2
        internal_file = target_entry.get("internal_result_file")
        if not internal_file:
            print(
                "AECMOS: the target result entry has no internal_result_file; "
                "re-run the benchmark in --split mode to record one."
            )
            return 2
        results_path, err = resolve_internal_ref(internal_file)
        if err:
            print(f"AECMOS: {err}")
            print("Re-run the benchmark in --split mode for this run.")
            return 2
        print(
            f"AECMOS: split manifest {manifest_path.name}, run "
            f"'{target_entry.get('run')}' (set {target_entry.get('set')})"
        )
    else:
        results_path = args.results
        if results_path is None:
            results_path = newest_results_file()
            if results_path is None:
                print(
                    "AECMOS: no internal result found under "
                    f"{DEFAULT_RESULTS_DIR}; run the internal benchmark first:"
                )
                print("  cargo run --release --example benchmark -- data\\test_set_icassp2022")
                return 2
        elif not results_path.is_absolute():
            results_path = CRATE_ROOT / results_path
    with open(results_path, "r", encoding="utf-8") as fh:
        internal = json.load(fh)
    clips = internal.get("clips", [])
    if not clips:
        print(f"AECMOS: {results_path} lists no clips; nothing to score.")
        return 2
    if args.limit and args.limit > 0:
        clips = stratified_clip_limit(clips, args.limit)

    # The measurement protocol of this internal result. A mix of aligned and
    # legacy clip records is inconsistent metadata and is refused outright.
    with_triplet = [bool(c.get("aligned")) for c in clips]
    if any(with_triplet) and not all(with_triplet):
        print(
            "AECMOS: REFUSED, the internal result mixes aligned (protocol 2) "
            "and legacy clip records; not scoring any of it."
        )
        return 3
    protocol = 2 if all(with_triplet) else 1
    if protocol == 1:
        if args.condition == "raw":
            print(
                "AECMOS: REFUSED, the raw condition needs a protocol 2 result "
                "with canonical aligned triplets; this result is legacy "
                "protocol 1."
            )
            return 3
        print(
            "AECMOS: legacy protocol 1 result (no aligned triplets); scoring "
            "with the original load-and-resample path."
        )

    set_name = internal.get("input", {}).get("set_name", "set")
    source_sha = internal.get("source", {}).get("combined_sha256", "")

    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # Resume from a checkpoint tied to this internal result, model, and
    # condition, so an interrupted run keeps its completed clips; it is
    # removed on a clean finish.
    ckpt_path = checkpoint_path(results_path, args.condition)
    done = load_checkpoint(ckpt_path, results_path.name, model_sha, args.condition)
    pending = [c for c in clips if c["id"] not in done]

    print(f"AECMOS: {model_path.name} ({params['note']}), model sha-256 {model_sha[:16]}")
    print(
        f"  scoring {len(clips)} clips from {results_path.name} "
        f"(protocol {protocol}, condition {args.condition}, "
        f"{len(done)} already done, {len(pending)} to score, "
        f"{args.workers} worker(s))"
    )

    notes = []
    since_ckpt = 0

    def handle(result):
        nonlocal since_ckpt
        entry, note = result
        if note:
            notes.append(note)
        if entry is None:
            return
        done[entry["id"]] = entry
        print(
            f"  {entry['scenario']:<20} {entry['id'][:44]:<46} "
            f"echo {entry['echo_mos']:5.2f}  deg {entry['deg_mos']:5.2f}"
        )
        since_ckpt += 1
        if since_ckpt >= CHECKPOINT_EVERY:
            save_checkpoint(ckpt_path, results_path.name, model_sha, args.condition, done)
            since_ckpt = 0

    if pending:
        try:
            if args.workers and args.workers > 1:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=args.workers,
                    initializer=worker_init,
                    initargs=(str(model_path), args.worker_threads, args.condition),
                ) as pool:
                    for result in pool.imap_unordered(worker_score, pending, chunksize=1):
                        handle(result)
            else:
                worker_init(str(model_path), args.worker_threads, args.condition)
                for clip in pending:
                    handle(worker_score(clip))
        except ValueError as e:
            if str(e).startswith(TRIPLET_PREFIX):
                save_checkpoint(
                    ckpt_path, results_path.name, model_sha, args.condition, done
                )
                print()
                print(f"AECMOS: REFUSED record, run aborted. {str(e)[len(TRIPLET_PREFIX):]}")
                print("  Nothing was scored for the refused record and no result was written.")
                return 3
            raise
        save_checkpoint(ckpt_path, results_path.name, model_sha, args.condition, done)

    # The scored list in the internal result's clip order.
    scored = [done[c["id"]] for c in clips if c["id"] in done]
    if not scored:
        print("AECMOS: nothing scored (all clips skipped).")
        return 2

    lines = []
    lines.append("AECMOS summary (EchoMOS and DegMOS per scenario, both AECMOS")
    lines.append("reference outputs ranging 1 to 5). Columns below are, for each")
    lines.append("scenario, the mean, median, and minimum of EchoMOS and then of")
    lines.append("DegMOS:")
    lines.append(
        f"  {'scenario':<20} {'clips':>5} {'echo mean':>10} {'echo med':>9} "
        f"{'echo min':>9} {'deg mean':>9} {'deg med':>8} {'deg min':>8}"
    )
    summary = {}
    for scenario in ("farend-singletalk", "doubletalk", "nearend-singletalk"):
        rows = [c for c in scored if c["scenario"] == scenario]
        if not rows:
            continue
        echo = sorted(c["echo_mos"] for c in rows)
        deg = sorted(c["deg_mos"] for c in rows)
        med = lambda v: v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
        summary[scenario] = {
            "clips": len(rows),
            "echo_mos": {
                "mean": round(sum(echo) / len(echo), 4),
                "median": round(med(echo), 4),
                "min": round(echo[0], 4),
            },
            "deg_mos": {
                "mean": round(sum(deg) / len(deg), 4),
                "median": round(med(deg), 4),
                "min": round(deg[0], 4),
            },
        }
        s = summary[scenario]
        lines.append(
            f"  {scenario:<20} {len(rows):>5} {s['echo_mos']['mean']:>10.2f} "
            f"{s['echo_mos']['median']:>9.2f} {s['echo_mos']['min']:>9.2f} "
            f"{s['deg_mos']['mean']:>9.2f} {s['deg_mos']['median']:>8.2f} "
            f"{s['deg_mos']['min']:>8.2f}"
        )
    for note in notes:
        lines.append(f"  note: {note}")

    stamp = datetime.now(timezone.utc)
    created = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    compact = stamp.strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schema": "decibri-aec-bench/aecmos/v2",
        "protocol": protocol,
        "condition": args.condition,
        "resampling": (
            "none (canonical aligned triplet)"
            if protocol == 2
            else "librosa on-load resample (legacy protocol 1, uncompensated "
            "resampler latency)"
        ),
        "created_utc": created,
        "model": {
            "file": model_path.name,
            "sha256": model_sha,
            "params": params["note"],
            "sample_rate_hz": sr,
        },
        "runtime": {
            "onnxruntime": ort.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
        "scored_portion_rule": {
            "st": "last half of the clip",
            "dt": "last (length_in_seconds - 15) / 2 seconds",
            "nst": "whole clip",
            "cap_seconds": MAX_SCORED_SECONDS,
        },
        "internal_result": {
            "file": results_path.name,
            "source_combined_sha256": source_sha,
            "set_name": set_name,
        },
        # How many clips this run scored, and the --limit that selected them
        # (null when every clip of the internal result was scored).
        "clips_scored": len(scored),
        "clip_limit": args.limit if args.limit and args.limit > 0 else None,
        "clips": scored,
        "scenarios": summary,
        "notes": notes,
    }

    # Every artifact lands beside the internal result it scored. In a run folder
    # that is a fixed name; in the --out-root layout the stamp and set name go
    # in the file name, because that folder holds many runs.
    out_dir = results_path.parent
    guard_out_dir(out_dir)
    prefix = "aecmos" if args.condition == "decibri" else "aecmos-raw"
    stem = prefix if in_run_dir(results_path) else f"{prefix}-{compact}-{set_name}"
    json_path = out_dir / f"{stem}.json"
    text_path = out_dir / f"{stem}.txt"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print(f"results: {json_path}")
    print(f"         {text_path}")

    # Split mode: fill the manifest entry's aecmos field with the per-scenario
    # means, in the manifest's own scenario order, leaving all else untouched.
    # Only the decibri condition fills the manifest; a raw-condition run is a
    # standalone baseline and never touches it.
    if manifest_path is not None and args.condition != "decibri":
        print()
        print("manifest: not touched (raw condition, standalone result only)")
    elif manifest_path is not None:
        aecmos_by_scenario = {}
        for scenario in ("doubletalk", "farend-singletalk", "nearend-singletalk"):
            if scenario in summary:
                aecmos_by_scenario[scenario] = {
                    "echo_mos": summary[scenario]["echo_mos"]["mean"],
                    "deg_mos": summary[scenario]["deg_mos"]["mean"],
                }
        ok, err = splice_aecmos(manifest_path, target_entry, aecmos_by_scenario)
        if ok:
            print()
            print(
                f"manifest: filled aecmos for run '{target_entry.get('run')}' "
                f"(set {target_entry.get('set')}) in {manifest_path}"
            )
        else:
            print()
            print(f"manifest: NOT updated ({err})")
            return 2

    # Clean finish: the final result is written, so drop the resume checkpoint.
    try:
        ckpt_path.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
