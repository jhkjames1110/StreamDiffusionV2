#!/usr/bin/env python3
"""
Online streaming benchmark for StreamDiffusionV2.

This measures the latency/throughput characteristics that matter for the
*online* (interactive, real-time) setting described in the project README
(`Online Inference (Web UI)`):

    * TTFF          - time-to-first-frame (stream start -> first generated frame)
    * FPS           - generated frames per second (what the viewer sees)
    * P99 latency   - 99th-percentile per-frame end-to-end latency
    * jitter        - frame-to-frame instability (latency jitter + cadence jitter)
    * SLO miss rate - fraction of frames slower than a latency budget

How it works
------------
The web demo (`demo/main.py`) wraps a serving object (`vid2vid.Pipeline` for a
single GPU, `vid2vid_pipe.MultiGPUPipeline` for pipeline-parallel) and talks to
it through exactly two calls:

    pipeline.accept_new_params(params)   # push one input frame (+ timestamp)
    pipeline.produce_outputs()           # drain ready output frames

This benchmark drives that *same* serving object directly. It reproduces the
demo's input pacing (a camera capped at `--input-fps`, default 16) and the
demo's metric definitions verbatim (`demo/main.py::produce_predictions` and
`App._log_metrics_to_file`): a FIFO match of input timestamps to output frames,
deadline-miss rate against `--target-latency`, and jitter as the variation of
consecutive latencies. Driving the object directly (instead of the
HTTP/WebSocket transport) isolates the pipeline's real serving cost without the
fragile browser/MJPEG layer; the transport adds only network-dependent overhead
that is not what we want to attribute to the model.

To benchmark the full HTTP/WebSocket transport instead, launch
`demo/main.py --enable-metrics --target-latency <s>` and read
`/api/metrics/{user_id}` (the same numbers, plus socket overhead).

Usage
-----
    # auto-detects an available checkpoint (1.3B if present, else 14B)
    python local/benchmark.py

    # explicit single-GPU 14B run for 30s at a 16fps camera, 1s SLO budget
    python local/benchmark.py \
        --config_path StreamDiffusionV2/configs/wan_causal_dmd_v2v_14b.yaml \
        --checkpoint_folder StreamDiffusionV2/ckpts/wan_causal_dmd_v2v_14b \
        --model_type T2V-14B --num_gpus 1 --gpu_ids 0 \
        --duration 30 --input-fps 16 --target-latency 1.0

    # pipeline-parallel across 2 GPUs
    python local/benchmark.py --num_gpus 2 --gpu_ids 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# --------------------------------------------------------------------------- #
# Repo / import wiring
# --------------------------------------------------------------------------- #
def find_repo_root(explicit: str | None) -> Path:
    """Locate the StreamDiffusionV2 source tree (the dir holding demo/ + streamv2v/)."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    here = Path(__file__).resolve().parent
    candidates += [
        here.parent / "StreamDiffusionV2",  # local/ is sibling of the source tree
        here / "StreamDiffusionV2",
        here.parent,
        here,
    ]
    for c in candidates:
        if (c / "demo" / "vid2vid.py").exists() and (c / "streamv2v").is_dir():
            return c
    raise FileNotFoundError(
        "Could not locate the StreamDiffusionV2 source tree (a directory containing "
        "demo/vid2vid.py and streamv2v/). Pass it explicitly with --repo."
    )


def resolve_model_defaults(repo: Path) -> dict:
    """Pick a config/checkpoint/model_type triple that actually exists on disk."""
    options = [
        ("configs/wan_causal_dmd_v2v.yaml", "ckpts/wan_causal_dmd_v2v", "T2V-1.3B"),
        ("configs/wan_causal_dmd_v2v_14b.yaml", "ckpts/wan_causal_dmd_v2v_14b", "T2V-14B"),
    ]
    for cfg, ckpt, model_type in options:
        if (repo / ckpt).exists():
            return {
                "config_path": str(repo / cfg),
                "checkpoint_folder": str(repo / ckpt),
                "model_type": model_type,
            }
    # Nothing downloaded yet: default to the 1.3B paths so the error is actionable.
    cfg, ckpt, model_type = options[0]
    return {
        "config_path": str(repo / cfg),
        "checkpoint_folder": str(repo / ckpt),
        "model_type": model_type,
    }


# --------------------------------------------------------------------------- #
# Input frames
# --------------------------------------------------------------------------- #
def load_video_frames(video_path: str, max_frames: int):
    """Decode up to `max_frames` RGB PIL frames from a video file (PyAV)."""
    import av
    from PIL import Image

    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(Image.fromarray(frame.to_rgb().to_ndarray()))
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return frames


def synthetic_frames(n: int, size: int):
    """Generate moving synthetic RGB frames (used when no video is available)."""
    from PIL import Image

    frames = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(n):
        phase = 2 * np.pi * i / n
        r = (0.5 + 0.5 * np.sin(xx / size * 6 + phase)) * 255
        g = (0.5 + 0.5 * np.sin(yy / size * 6 + phase * 1.3)) * 255
        b = (0.5 + 0.5 * np.sin((xx + yy) / size * 6 + phase * 0.7)) * 255
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


def build_frame_pool(args) -> list:
    """Return a list of PIL frames to stream (cyclically) into the pipeline."""
    pool_size = max(64, int(args.input_fps if args.input_fps > 0 else 16) * 8)
    if args.video and os.path.exists(args.video):
        frames = load_video_frames(args.video, pool_size)
        print(f"[input] loaded {len(frames)} frames from {args.video}")
        return frames
    if args.video:
        print(f"[input] WARNING: --video {args.video!r} not found; using synthetic frames")
    frames = synthetic_frames(pool_size, args.frame_size)
    print(f"[input] generated {len(frames)} synthetic {args.frame_size}x{args.frame_size} frames")
    return frames


# --------------------------------------------------------------------------- #
# Pipeline construction (faithful to demo/main.py)
# --------------------------------------------------------------------------- #
def build_pipeline_args(args):
    """Build the same `Args` NamedTuple demo/config.py produces, without argparse side effects."""
    # demo/config.py runs argparse at import time; neutralise argv so importing it is safe.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        from config import Args  # import inside function is intentional (after sys.path setup)
    finally:
        sys.argv = saved_argv

    return Args(
        host="0.0.0.0",
        port=7860,
        max_queue_size=0,
        timeout=0.0,
        ssl_certfile=None,
        ssl_keyfile=None,
        config_path=os.path.abspath(args.config_path),
        checkpoint_folder=os.path.abspath(args.checkpoint_folder),
        step=args.step,
        noise_scale=args.noise_scale,
        debug=False,
        num_gpus=args.num_gpus,
        gpu_ids=args.gpu_ids,
        max_outstanding=args.max_outstanding,
        schedule_block=args.schedule_block,
        model_type=args.model_type,
        use_taehv=args.use_taehv,
        use_tensorrt=args.use_tensorrt,
        fast=args.fast,
        enable_metrics=False,  # this script computes metrics itself
        target_latency=args.target_latency,
        t2v=False,
        online_batching_mode=args.online_batching_mode,
        online_slo_wait_threshold=args.online_slo_wait_threshold,
    )


def _patch_startup_timeout(module, seconds: float):
    """Raise the worker-readiness timeout (the 180s default is bound at def-time in vid2vid).

    Loading the 14B checkpoint on every rank concurrently in multi-GPU mode can take
    longer than 180s; this widens the window without touching the repo.
    """
    import functools

    base = getattr(module.wait_for_processes_ready, "__wrapped_base__", module.wait_for_processes_ready)
    patched = functools.partial(base, timeout_seconds=seconds)
    patched.__wrapped_base__ = base
    module.wait_for_processes_ready = patched


def construct_pipeline(pipeline_args, startup_timeout: float = 900.0):
    """Instantiate the single- or multi-GPU serving object exactly like demo/main.py."""
    from config import validate_online_batching_config

    validate_online_batching_config(pipeline_args.num_gpus, pipeline_args.online_batching_mode)
    if pipeline_args.num_gpus > 1:
        import vid2vid_pipe
        from vid2vid_pipe import MultiGPUPipeline

        _patch_startup_timeout(vid2vid_pipe, startup_timeout)  # prepare() calls the module-local name
        return MultiGPUPipeline(pipeline_args)
    import vid2vid
    from vid2vid import Pipeline

    _patch_startup_timeout(vid2vid, startup_timeout)
    return Pipeline(pipeline_args)


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
class OnlineBenchmark:
    def __init__(self, pipeline, frames, args):
        self.pipeline = pipeline
        self.frames = frames
        self.args = args

        self._stop = threading.Event()
        self._lock = threading.Lock()

        # FIFO of input feed timestamps (matched to outputs, like demo/main.py).
        self._input_ts = deque()

        # Per-frame results.
        self.latencies: list[float] = []         # end-to-end input->output latency (s)
        self.output_times: list[float] = []      # arrival perf_counter() of each output frame
        self.batch_sizes: list[int] = []         # frames returned per produce_outputs() call

        self.first_input_t: float | None = None
        self.first_output_t: float | None = None
        self.fed = 0

        # Optional: keep generated frames in memory (uint8) to write an mp4 at the
        # end. Buffered, never written on the hot path, so it does not perturb timing.
        self.save_enabled = bool(args.save_video)
        self.max_save = args.max_save_frames
        self.saved_frames: list[np.ndarray] = []

    # ---- producer: feed frames at the camera rate, with optional back-pressure ----
    def _producer(self):
        prompt = self.args.prompt
        period = 1.0 / self.args.input_fps if self.args.input_fps > 0 else 0.0
        next_t = time.perf_counter()
        i = 0
        while not self._stop.is_set():
            # Back-pressure: don't let more than --max-pending unconsumed frames pile up.
            if self.args.max_pending > 0:
                while (
                    not self._stop.is_set()
                    and self.pipeline.input_queue.qsize() >= self.args.max_pending
                ):
                    time.sleep(0.002)
                if self._stop.is_set():
                    break

            # Rate cap (camera FPS).
            if period > 0:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now)
                next_t += period

            frame = self.frames[i % len(self.frames)]
            i += 1
            # Only send the prompt on the very first frame; changing it mid-stream
            # would trigger a session restart in the worker.
            params = SimpleNamespace(image=frame)
            if self.fed == 0:
                params.prompt = prompt

            t = time.perf_counter()
            with self._lock:
                if self.first_input_t is None:
                    self.first_input_t = t
                self._input_ts.append(t)
                self.fed += 1
            self.pipeline.accept_new_params(params)

    # ---- collector: drain outputs and FIFO-match latencies (like produce_predictions) ----
    def _collector(self):
        while not self._stop.is_set():
            images = self.pipeline.produce_outputs()
            if not images:
                time.sleep(1.0 / 240)
                continue
            out_t = time.perf_counter()
            with self._lock:
                if self.first_output_t is None:
                    self.first_output_t = out_t
                self.batch_sizes.append(len(images))
                for img in images:
                    self.output_times.append(out_t)
                    if self._input_ts:
                        self.latencies.append(out_t - self._input_ts.popleft())
                    if self.save_enabled and len(self.saved_frames) < self.max_save:
                        self.saved_frames.append(np.asarray(img, dtype=np.uint8))

    def run(self):
        producer = threading.Thread(target=self._producer, name="producer", daemon=True)
        collector = threading.Thread(target=self._collector, name="collector", daemon=True)
        collector.start()
        producer.start()

        deadline = time.perf_counter() + self.args.duration
        try:
            while time.perf_counter() < deadline:
                with self._lock:
                    n = len(self.output_times)
                if self.args.frames > 0 and n >= self.args.frames:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[run] interrupted; stopping early")

        self._stop.set()
        producer.join(timeout=5)
        collector.join(timeout=5)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def compute_metrics(bench: OnlineBenchmark, args) -> dict:
    out_times = np.array(bench.output_times, dtype=np.float64)
    latencies_all = np.array(bench.latencies, dtype=np.float64)
    n_out = out_times.size

    result: dict = {
        "config": {
            "model_type": args.model_type,
            "num_gpus": args.num_gpus,
            "gpu_ids": args.gpu_ids,
            "step": args.step,
            "online_batching_mode": args.online_batching_mode,
            "input_fps_cap": args.input_fps,
            "max_pending": args.max_pending,
            "target_latency_s": args.target_latency,
            "warmup_frames": args.warmup,
            "duration_s": args.duration,
            "frames_fed": bench.fed,
            "frames_generated": n_out,
        }
    }

    # ---- TTFF ----
    if bench.first_input_t is not None and bench.first_output_t is not None:
        result["ttff_s"] = bench.first_output_t - bench.first_input_t
    else:
        result["ttff_s"] = None

    if n_out == 0:
        result["error"] = "No frames were generated within the run window."
        return result

    # ---- FPS ----
    span_steady = out_times[-1] - out_times[0]
    result["fps"] = {
        # frames after the first, over the wall time between first and last frame
        "steady_state": float((n_out - 1) / span_steady) if span_steady > 0 else float("nan"),
        # everything since the stream started (includes TTFF warmup gap)
        "end_to_end": (
            float(n_out / (out_times[-1] - bench.first_input_t))
            if bench.first_input_t is not None and out_times[-1] > bench.first_input_t
            else float("nan")
        ),
        "frames": n_out,
        "wall_span_s": float(span_steady),
    }

    # ---- steady-state windows (exclude warmup) ----
    w = min(args.warmup, max(0, n_out - 1))
    lat = latencies_all[w:] if latencies_all.size > w else latencies_all
    out_w = out_times[w:]

    # ---- latency distribution ----
    if lat.size:
        result["latency_s"] = {
            "mean": float(np.mean(lat)),
            "p50": percentile(lat, 50),
            "p90": percentile(lat, 90),
            "p95": percentile(lat, 95),
            "p99": percentile(lat, 99),
            "p99_9": percentile(lat, 99.9),
            "min": float(np.min(lat)),
            "max": float(np.max(lat)),
            "std": float(np.std(lat)),
            "samples": int(lat.size),
        }

        # ---- SLO / deadline miss rate (latency > target) ----
        missed = int(np.sum(lat > args.target_latency))
        result["slo"] = {
            "target_latency_s": args.target_latency,
            "miss_rate": float(missed / lat.size),
            "missed_frames": missed,
            "total_frames": int(lat.size),
        }

        # ---- latency jitter: variation of consecutive latencies (demo/main.py def) ----
        if lat.size > 1:
            jd = np.abs(np.diff(lat))
            result["latency_jitter_s"] = {
                "mean": float(np.mean(jd)),
                "std": float(np.std(jd)),
                "p95": percentile(jd, 95),
                "p99": percentile(jd, 99),
                "max": float(np.max(jd)),
            }

    # ---- cadence jitter: variation of output inter-arrival gaps (display smoothness) ----
    if out_w.size > 2:
        gaps = np.diff(out_w)
        result["cadence_s"] = {
            "mean_interval": float(np.mean(gaps)),
            "std_interval": float(np.std(gaps)),
            "p95_interval": percentile(gaps, 95),
            "p99_interval": percentile(gaps, 99),
            "max_interval": float(np.max(gaps)),
            # inter-arrival jitter (RFC-3550 style): mean |gap_i - gap_{i-1}|
            "interarrival_jitter": float(np.mean(np.abs(np.diff(gaps)))) if gaps.size > 1 else 0.0,
        }

    # ---- batching behaviour (insight into stream-batch vs wo_batch) ----
    if bench.batch_sizes:
        bs = np.array(bench.batch_sizes)
        result["output_batch"] = {
            "mean_frames_per_drain": float(np.mean(bs)),
            "max_frames_per_drain": int(np.max(bs)),
            "drains": int(bs.size),
        }

    return result


def print_report(m: dict) -> None:
    c = m["config"]
    line = "=" * 68
    print("\n" + line)
    print("StreamDiffusionV2 - ONLINE STREAMING BENCHMARK")
    print(line)
    print(f"  model={c['model_type']}  gpus={c['num_gpus']} ({c['gpu_ids']})  "
          f"step={c['step']}  batching={c['online_batching_mode']}")
    print(f"  input cap={c['input_fps_cap']} fps  max_pending={c['max_pending']}  "
          f"SLO budget={c['target_latency_s']}s  warmup={c['warmup_frames']}")
    print(f"  fed {c['frames_fed']} frames, generated {c['frames_generated']} frames "
          f"in {c['duration_s']}s window")
    print(line)

    if "error" in m:
        print(f"  ERROR: {m['error']}")
        print(line + "\n")
        return

    ttff = m.get("ttff_s")
    print(f"  TTFF (time to first frame) : {ttff*1000:8.1f} ms" if ttff is not None
          else "  TTFF (time to first frame) :      n/a")

    fps = m["fps"]
    print(f"  FPS (steady-state)         : {fps['steady_state']:8.2f}  "
          f"(end-to-end {fps['end_to_end']:.2f})")

    if "latency_s" in m:
        L = m["latency_s"]
        print(f"  Latency mean / p50         : {L['mean']*1000:8.1f} / {L['p50']*1000:.1f} ms")
        print(f"  Latency p95 / P99          : {L['p95']*1000:8.1f} / {L['p99']*1000:.1f} ms")
        print(f"  Latency p99.9 / max        : {L['p99_9']*1000:8.1f} / {L['max']*1000:.1f} ms")

    if "slo" in m:
        s = m["slo"]
        print(f"  SLO miss rate (>{s['target_latency_s']}s)      : {s['miss_rate']*100:8.2f} %  "
              f"({s['missed_frames']}/{s['total_frames']})")

    if "latency_jitter_s" in m:
        j = m["latency_jitter_s"]
        print(f"  Latency jitter mean / p99  : {j['mean']*1000:8.1f} / {j['p99']*1000:.1f} ms")
    if "cadence_s" in m:
        cd = m["cadence_s"]
        print(f"  Cadence interval mean/std  : {cd['mean_interval']*1000:8.1f} / "
              f"{cd['std_interval']*1000:.1f} ms  (jitter {cd['interarrival_jitter']*1000:.1f} ms)")
    if "output_batch" in m:
        ob = m["output_batch"]
        print(f"  Output drain batch (mean)  : {ob['mean_frames_per_drain']:8.2f} frames "
              f"(max {ob['max_frames_per_drain']})")
    print(line + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Online streaming benchmark for StreamDiffusionV2 "
                    "(TTFF / FPS / P99 latency / jitter / SLO miss rate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # model / pipeline (defaults auto-detected from what is on disk)
    p.add_argument("--repo", default=None, help="Path to the StreamDiffusionV2 source tree")
    p.add_argument("--config_path", default=None, help="Model config YAML (auto-detected if unset)")
    p.add_argument("--checkpoint_folder", default=None, help="Checkpoint dir (auto-detected if unset)")
    p.add_argument("--model_type", default=None, choices=["T2V-1.3B", "T2V-14B"],
                   help="Model layout (auto-detected if unset)")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpu_ids", default=None, help="Comma-separated GPU ids (defaults to 0..num_gpus-1)")
    p.add_argument("--step", type=int, default=2, help="Denoising steps (2 = real-time sweet spot)")
    p.add_argument("--noise_scale", type=float, default=0.8)
    p.add_argument("--online_batching_mode", choices=["batch", "wo_batch", "auto"], default="batch")
    p.add_argument("--online_slo_wait_threshold", type=float, default=0.5)
    p.add_argument("--use_taehv", action="store_true", help="Lightweight TAEHV VAE decoder")
    p.add_argument("--use_tensorrt", action="store_true")
    p.add_argument("--fast", action="store_true", help="Enable TAEHV + TensorRT fast path")
    p.add_argument("--max_outstanding", type=int, default=2, help="Multi-GPU in-flight sends/recvs")
    p.add_argument("--schedule_block", action="store_true", help="Multi-GPU block scheduling")
    p.add_argument("--startup-timeout", dest="startup_timeout", type=float, default=900.0,
                   help="Seconds to wait for worker(s) to load before giving up")

    # workload / load generation
    p.add_argument("--prompt", default="A pug walks on the grass, realistic")
    p.add_argument("--video", default=None,
                   help="Input video to stream (default: <repo>/examples/original.mp4 if present)")
    p.add_argument("--frame_size", type=int, default=512, help="Synthetic frame size (fallback only)")
    p.add_argument("--input-fps", dest="input_fps", type=float, default=16.0,
                   help="Camera input rate cap; 0 = feed as fast as the pipeline drains")
    p.add_argument("--max-pending", dest="max_pending", type=int, default=8,
                   help="Back-pressure: max unconsumed input frames; 0 = unbounded (open-loop)")
    p.add_argument("--duration", type=float, default=30.0, help="Measurement window in seconds")
    p.add_argument("--frames", type=int, default=0, help="Stop after N generated frames (0 = use --duration)")
    p.add_argument("--warmup", type=int, default=8, help="Output frames excluded from steady-state stats")

    # reporting
    p.add_argument("--target-latency", dest="target_latency", type=float, default=1.0,
                   help="SLO / deadline budget in seconds for the miss-rate metric")
    p.add_argument("--output-json", dest="output_json", default=None, help="Write metrics JSON here")
    p.add_argument("--save-video", dest="save_video", default=None,
                   help="Write the generated frames to this mp4 (buffered, written after the run)")
    p.add_argument("--save-fps", dest="save_fps", type=int, default=16,
                   help="Playback fps for the saved mp4")
    p.add_argument("--max-save-frames", dest="max_save_frames", type=int, default=600,
                   help="Cap on frames buffered for --save-video (limits memory)")
    return p.parse_args()


def main():
    args = parse_args()

    repo = find_repo_root(args.repo)
    # Make demo/ and the repo root importable (vid2vid, config, streamv2v).
    sys.path.insert(0, str(repo / "demo"))
    sys.path.insert(0, str(repo))

    # Fill in model defaults from whatever checkpoints exist.
    defaults = resolve_model_defaults(repo)
    args.config_path = args.config_path or defaults["config_path"]
    args.checkpoint_folder = args.checkpoint_folder or defaults["checkpoint_folder"]
    args.model_type = args.model_type or defaults["model_type"]
    if args.gpu_ids is None:
        args.gpu_ids = ",".join(str(i) for i in range(args.num_gpus))
    if args.video is None:
        default_vid = repo / "examples" / "original.mp4"
        args.video = str(default_vid) if default_vid.exists() else ""
    if args.fast:
        args.use_taehv = True
        args.use_tensorrt = True

    print(f"[setup] repo={repo}")
    print(f"[setup] config={args.config_path}")
    print(f"[setup] checkpoint={args.checkpoint_folder}")
    print(f"[setup] model_type={args.model_type}  num_gpus={args.num_gpus}  gpu_ids={args.gpu_ids}")
    if not os.path.exists(args.checkpoint_folder):
        print(f"[setup] WARNING: checkpoint folder does not exist: {args.checkpoint_folder}")

    # CUDA + multiprocessing must use spawn (matches demo/main.py).
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    frames = build_frame_pool(args)

    pipeline_args = build_pipeline_args(args)

    print("[setup] loading model / starting worker(s) ... (this can take a few minutes for 14B)")
    t0 = time.perf_counter()
    pipeline = construct_pipeline(pipeline_args, startup_timeout=args.startup_timeout)
    print(f"[setup] pipeline ready in {time.perf_counter() - t0:.1f}s; starting benchmark\n")

    bench = OnlineBenchmark(pipeline, frames, args)
    metrics = {}
    try:
        bench.run()
        metrics = compute_metrics(bench, args)
        print_report(metrics)
    finally:
        try:
            pipeline.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[teardown] pipeline.close() raised: {exc}")

    if args.output_json and metrics:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[output] metrics written to {args.output_json}")

    if args.save_video and bench.saved_frames:
        from streamdiffusionv2 import export_video  # lazy: heavy import

        video = np.stack(bench.saved_frames, axis=0).astype(np.float32) / 255.0
        export_video(video, args.save_video, fps=args.save_fps)
        print(f"[output] saved {len(bench.saved_frames)} generated frames -> {args.save_video}")
    elif args.save_video:
        print("[output] --save-video requested but no frames were generated")


if __name__ == "__main__":
    main()
    # The mp.Manager process and Queue feeder threads can keep the interpreter
    # alive after the report is written; force a clean, immediate exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
