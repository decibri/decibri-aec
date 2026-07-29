# benchmarks/

Benchmarking kit for decibri-aec. It measures the shipped engine through its
public API only, then scores the result with AECMOS, the objective metric used
to rank submissions in the Microsoft AEC Challenges.

The dataset and the scoring model are third-party material. They are not shipped,
fetched, or redistributed by this kit. See `ATTRIBUTIONS.md` for their origin,
their licence terms, and the citations any published number must carry.

## What you need

- A Rust toolchain, to build and run the engine benchmark.
- The Microsoft AEC-Challenge ICASSP 2022 test set, unpacked at
  `data/test_set_icassp2022/` as three scenario folders (`doubletalk/`,
  `farend-singletalk/`, `nearend-singletalk/`), each holding `<stem>_mic.wav`
  and `<stem>_lpb.wav` pairs.
- The AECMOS scoring model at `models/Run_1663915512_Stage_0.onnx`.
- Python 3 with `numpy`, `librosa`, and `onnxruntime`, for the scoring step.

`data/` and `models/` are gitignored, and nothing here downloads them for you.

## Environment for the scorer

The engine benchmark needs only cargo. The scoring step needs a small Python
environment with three packages. One portable way to build it is

```sh
python -m venv benchmarks/.venv
benchmarks/.venv/Scripts/python.exe -m pip install numpy librosa onnxruntime
```

On Linux or macOS the interpreter is `benchmarks/.venv/bin/python`. Below,
`python` means that environment's interpreter.

## Reproducing the published numbers

The performance table in the top-level `README.md` reports, per scenario, the
mean AECMOS EchoMOS and DegMOS of the microphone before cancellation and of
decibri's output after it, over the 800-clip pool. Three commands reproduce it.

Run the engine over the pool. This writes one self-contained run directory
holding the internal metrics (`bench.json`), the enhanced output, and a
canonical aligned triplet per clip.

```sh
cargo run --release --example benchmark -- data/test_set_icassp2022 --run-dir data/runs/pool
```

Score decibri's output, the "after" column.

```sh
python benchmarks/run_aecmos.py --results data/runs/pool/bench.json
```

Score the raw microphone, the "before" column. This scores the same run again
with the microphone in place of the enhanced signal.

```sh
python benchmarks/run_aecmos.py --results data/runs/pool/bench.json --condition raw
```

Each scoring step prints a per-scenario table and writes it beside the run as
`aecmos.json` / `aecmos.txt` for decibri and `aecmos-raw.json` / `aecmos-raw.txt`
for raw. The echo-mean and deg-mean columns are the published EchoMOS and
DegMOS. `--run-dir` names any directory under `data/`; the commands above simply
point at the same one.

## Why before and after are comparable

Both columns come from one engine run. For each clip the benchmark writes a
canonical aligned triplet, the microphone, the loopback reference, and the
enhanced output, all at one sample rate and one length on a single timeline. The
"after" score reads the enhanced signal of that triplet, and the "before" score
reads the microphone of the same triplet. Nothing else differs between the two,
so the change in score is the work of the canceller and nothing else.

The kit verifies that agreement before it trusts a comparison. The two scored
sets must name the same internal benchmark result, cover the same clips, use the
same AECMOS model, and score the same region of each clip, all under the same
measurement protocol. If any of those disagree the kit refuses to report a
number rather than compare two measurements that are not the same.
