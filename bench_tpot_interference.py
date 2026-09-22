"""Measure decode TPOT when a long prompt arrives during active decoding.

Run the same command in the upstream baseline checkout and in this mixed-batch
checkout.  The script only drives the public ``LLM.add_request`` / ``LLM.step``
API, so the observed schedule is the checkout's real production schedule.

Example:

    python bench_tpot_interference.py --model /path/to/Qwen3-0.6B \
        --label chunked --output-dir results/tpot-interference/chunked
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch


@dataclass(frozen=True)
class IterationSample:
    label: str
    prompt_length: int
    iteration: int
    iteration_ms: float
    decode_tokens: int
    long_prefill_tokens: int
    long_prefill_progress: int
    decode_tpot_ms: float | None


@dataclass(frozen=True)
class ScenarioSummary:
    label: str
    prompt_length: int
    long_prefill_iterations: int
    decode_while_prefill_iterations: int
    zero_decode_iterations: int
    max_decode_tpot_ms: float
    p50_decode_tpot_ms: float
    p95_decode_tpot_ms: float
    long_prompt_finish_ms: float


def token_ids(seed: int, length: int) -> list[int]:
    """Create deterministic, distinct token streams without tokenizer work."""
    return [(seed + index * 31) % 32_000 for index in range(length)]


def percentile(values: list[float], ratio: float) -> float:
    assert values
    return sorted(values)[math.ceil(len(values) * ratio) - 1]


def cuda_step(llm: LLM) -> float:
    """Synchronize around one engine step so wall time includes GPU execution."""
    torch.cuda.synchronize()
    started = perf_counter()
    llm.step()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1_000


def run_scenario(
    llm: LLM,
    *,
    label: str,
    prompt_length: int,
    num_decode_requests: int,
    decode_prompt_length: int,
    decode_max_tokens: int,
    post_long_decode_steps: int,
) -> tuple[list[IterationSample], ScenarioSummary]:
    """Inject one long request after the short requests have entered RUNNING."""
    decode_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=decode_max_tokens)
    decode_seqs = []
    for request_index in range(num_decode_requests):
        llm.add_request(token_ids(1_000 + request_index * 1_000, decode_prompt_length), decode_params)
        decode_seqs.append(llm.scheduler.waiting[-1])

    cuda_step(llm)
    if any(seq.num_completion_tokens != 1 for seq in decode_seqs):
        raise RuntimeError("decode setup did not place every short request into RUNNING")

    long_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
    llm.add_request(token_ids(100_000 + prompt_length, prompt_length), long_params)
    long_seq = llm.scheduler.waiting[-1]

    samples: list[IterationSample] = []
    decode_gap_ms = 0.0
    post_long_decode_steps_observed = 0
    long_finish_ms: float | None = None
    max_iterations = math.ceil(prompt_length / llm.scheduler.max_num_batched_tokens) + post_long_decode_steps + 8

    for iteration in range(1, max_iterations + 1):
        completion_before = [seq.num_completion_tokens for seq in decode_seqs]
        prefill_before = long_seq.num_cached_tokens
        long_was_finished = long_seq.is_finished

        iteration_ms = cuda_step(llm)
        decode_tokens = sum(
            seq.num_completion_tokens - before for seq, before in zip(decode_seqs, completion_before)
        )
        long_finished_this_step = long_seq.is_finished and not long_was_finished
        long_prefill_tokens = (
            long_seq.num_prompt_tokens - prefill_before
            if long_finished_this_step
            else long_seq.num_cached_tokens - prefill_before
        )
        long_prefill_progress = (
            long_seq.num_prompt_tokens if long_seq.is_finished else long_seq.num_cached_tokens
        )
        decode_gap_ms += iteration_ms
        decode_tpot_ms = None
        if decode_tokens:
            # Every original decode request advances at most once per iteration.
            decode_tpot_ms = decode_gap_ms
            decode_gap_ms = 0.0
            if long_was_finished:
                post_long_decode_steps_observed += 1

        if long_finish_ms is None and long_seq.is_finished:
            long_finish_ms = sum(sample.iteration_ms for sample in samples) + iteration_ms

        samples.append(
            IterationSample(
                label=label,
                prompt_length=prompt_length,
                iteration=iteration,
                iteration_ms=iteration_ms,
                decode_tokens=decode_tokens,
                long_prefill_tokens=long_prefill_tokens,
                long_prefill_progress=long_prefill_progress,
                decode_tpot_ms=decode_tpot_ms,
            )
        )

        if long_finish_ms is not None and post_long_decode_steps_observed >= post_long_decode_steps:
            break
    else:
        raise RuntimeError(
            "scenario did not observe the requested post-prefill decode steps: "
            f"long finished={long_seq.is_finished}, "
            f"long prefill {long_seq.num_cached_tokens}/{long_seq.num_prompt_tokens}, "
            f"post-long decode steps {post_long_decode_steps_observed}/{post_long_decode_steps}, "
            f"decode completion tokens {[seq.num_completion_tokens for seq in decode_seqs]}"
        )

    # Keep subsequent scenarios independent of outstanding running requests.
    while not llm.is_finished():
        cuda_step(llm)

    tpot_values = [sample.decode_tpot_ms for sample in samples if sample.decode_tpot_ms is not None]
    assert long_finish_ms is not None and tpot_values
    long_prefill_samples = [sample for sample in samples if sample.long_prefill_tokens]
    return samples, ScenarioSummary(
        label=label,
        prompt_length=prompt_length,
        long_prefill_iterations=len(long_prefill_samples),
        decode_while_prefill_iterations=sum(sample.decode_tokens > 0 for sample in long_prefill_samples),
        zero_decode_iterations=sum(sample.decode_tokens == 0 for sample in samples),
        max_decode_tpot_ms=max(tpot_values),
        p50_decode_tpot_ms=statistics.median(tpot_values),
        p95_decode_tpot_ms=percentile(tpot_values, 0.95),
        long_prompt_finish_ms=long_finish_ms,
    )


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local model directory")
    parser.add_argument("--label", required=True, help="result label, e.g. baseline or chunked")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for CSV results")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--num-decode-requests", type=int, default=16)
    parser.add_argument("--decode-prompt-length", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--post-long-decode-steps", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if min(args.prompt_lengths) <= 0:
        raise SystemExit("--prompt-lengths must be positive")
    if args.num_decode_requests <= 0 or args.decode_prompt_length <= 0:
        raise SystemExit("decode request count and prompt length must be positive")
    if args.max_num_batched_tokens < args.num_decode_requests * args.decode_prompt_length:
        raise SystemExit("batch budget must fit the initial short-request prefill")
    if args.max_num_batched_tokens <= args.num_decode_requests:
        raise SystemExit("batch budget must leave token capacity after decode-first scheduling")
    if args.max_model_len < max(args.prompt_lengths):
        raise SystemExit("--max-model-len must fit the longest long prompt")


def main() -> None:
    global LLM, SamplingParams
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("this performance experiment requires CUDA")
    from nanovllm import LLM, SamplingParams

    # The short requests need to remain alive through all chunks of the 16K case.
    decode_max_tokens = math.ceil(max(args.prompt_lengths) / args.max_num_batched_tokens) + args.post_long_decode_steps + 2
    llm = LLM(
        str(args.model),
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.num_decode_requests + 1,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )

    # Warm up compilation and CUDA-graph setup; its measurements are discarded.
    llm.add_request(token_ids(31, 8), SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1))
    while not llm.is_finished():
        cuda_step(llm)

    all_samples: list[IterationSample] = []
    summaries: list[ScenarioSummary] = []
    for prompt_length in args.prompt_lengths:
        samples, summary = run_scenario(
            llm,
            label=args.label,
            prompt_length=prompt_length,
            num_decode_requests=args.num_decode_requests,
            decode_prompt_length=args.decode_prompt_length,
            decode_max_tokens=decode_max_tokens,
            post_long_decode_steps=args.post_long_decode_steps,
        )
        all_samples.extend(samples)
        summaries.append(summary)
        print(
            f"{args.label} {prompt_length // 1024}K: "
            f"max TPOT {summary.max_decode_tpot_ms:.2f} ms, "
            f"p95 {summary.p95_decode_tpot_ms:.2f} ms, "
            f"decode during prefill {summary.decode_while_prefill_iterations}/"
            f"{summary.long_prefill_iterations} iterations"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "iterations.csv", all_samples)
    write_csv(args.output_dir / "summary.csv", summaries)
    print(f"wrote {args.output_dir / 'iterations.csv'}")
    print(f"wrote {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
