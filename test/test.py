"""GPU correctness suite for mixed-batch scheduling.

Run this on a CUDA host after syncing commit 15aa322 (or a descendant):

    python test/test.py --model /path/to/Qwen3-0.6B

The suite deliberately uses ``LLM.add_request()`` and ``LLM.step()``. Test
hooks only observe the plan and tensors for the current step; they do not build
``SchedulerOutput`` by hand or change production scheduling decisions.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import ScheduledSequence, Scheduler, SchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.context import get_context


SAMPLE_SEED = 20260921
MAX_BATCH_TOKENS = 16


def token_ids(start: int, length: int) -> list[int]:
    return list(range(start, start + length))


@dataclass(frozen=True)
class PlannedItem:
    """Observation captured before ``LLM.step()`` commits its plan."""

    seq: Sequence
    is_prefill: bool
    cached_tokens: int
    scheduled_tokens: int
    num_tokens: int
    last_token: int


def run_engine_step(llm: LLM, seed: int, observe_tensors: bool = False):
    """Run one public engine step and retain only evidence for this test step."""
    captured: dict[str, object] = {}
    original_schedule = llm.scheduler.schedule
    original_call = llm.model_runner.call
    original_run_model = llm.model_runner.run_model

    def capture_schedule():
        output = original_schedule()
        captured["plan"] = [
            PlannedItem(
                item.seq,
                item.is_prefill,
                item.seq.num_cached_tokens,
                item.seq.num_scheduled_tokens,
                item.seq.num_tokens,
                item.seq.last_token,
            )
            for item in output.scheduled
        ]
        return output

    def capture_call(method_name, *args):
        result = original_call(method_name, *args)
        if method_name == "run":
            captured["token_ids"] = result
        return result

    def capture_run_model(input_ids, positions, use_varlen):
        context = get_context()
        captured["run_model_calls"] = captured.get("run_model_calls", 0) + 1
        captured["use_varlen"] = use_varlen
        captured["input_ids"] = input_ids.tolist()
        captured["positions"] = positions.tolist()
        captured["cu_seqlens_q"] = None if context.cu_seqlens_q is None else context.cu_seqlens_q.tolist()
        captured["cu_seqlens_k"] = None if context.cu_seqlens_k is None else context.cu_seqlens_k.tolist()
        captured["slot_mapping_size"] = None if context.slot_mapping is None else context.slot_mapping.numel()
        captured["block_table_shape"] = None if context.block_tables is None else tuple(context.block_tables.shape)
        return original_run_model(input_ids, positions, use_varlen)

    llm.scheduler.schedule = capture_schedule
    llm.model_runner.call = capture_call
    if observe_tensors:
        llm.model_runner.run_model = capture_run_model
    try:
        torch.manual_seed(seed)
        outputs, _ = llm.step()
    finally:
        llm.scheduler.schedule = original_schedule
        llm.model_runner.call = original_call
        llm.model_runner.run_model = original_run_model
    return captured["plan"], captured, outputs


def make_running(llm: LLM, prompt: list[int], params: SamplingParams, seed: int) -> Sequence:
    """Submit a complete prompt and leave its first generated token in RUNNING."""
    llm.add_request(prompt, params)
    plan, captured, _ = run_engine_step(llm, seed)
    assert [item.is_prefill for item in plan] == [True]
    assert isinstance(captured["token_ids"][0], int)
    seq = plan[0].seq
    assert seq.status is SequenceStatus.RUNNING
    assert seq.num_cached_tokens == len(prompt)
    assert seq.num_completion_tokens == 1
    return seq


def drain(llm: LLM, seed: int) -> None:
    """Finish requests through the real engine, detecting a queue that gets stuck."""
    for step in range(64):
        if llm.is_finished():
            return
        run_engine_step(llm, seed + step)
    raise AssertionError("engine did not drain within 64 scheduling steps")


def assert_no_owned_kv_blocks(llm: LLM) -> None:
    assert llm.is_finished()
    assert not llm.scheduler.block_manager.used_block_ids


def run_pure_path_regression(llm: LLM) -> None:
    """Pure prefill and pure decode must retain their legacy execution modes."""
    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    llm.add_request(token_ids(100, 5), params)
    llm.add_request(token_ids(200, 7), params)

    plan, captured, _ = run_engine_step(llm, SAMPLE_SEED, observe_tensors=True)
    assert [item.is_prefill for item in plan] == [True, True]
    assert captured["use_varlen"] is True
    assert all(isinstance(token_id, int) for token_id in captured["token_ids"])

    plan, captured, outputs = run_engine_step(llm, SAMPLE_SEED + 1, observe_tensors=True)
    assert [item.is_prefill for item in plan] == [False, False]
    assert captured["use_varlen"] is False
    assert len(outputs) == 2
    assert_no_owned_kv_blocks(llm)
    print("PASS: pure prefill, pure decode, max_tokens finish, and KV release")


def run_complete_prefill_mixed_case(llm: LLM) -> None:
    """A decode and a fitting prompt share one varlen forward and both sample."""
    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    decode_seq = make_running(llm, token_ids(300, 6), params, SAMPLE_SEED + 10)
    llm.add_request(token_ids(400, 5), params)

    plan, captured, _ = run_engine_step(llm, SAMPLE_SEED + 11, observe_tensors=True)
    assert [item.is_prefill for item in plan] == [False, True]
    assert plan[0].seq is decode_seq
    assert sum(item.scheduled_tokens for item in plan) == 6
    assert captured["run_model_calls"] == 1
    assert captured["use_varlen"] is True
    assert all(isinstance(token_id, int) for token_id in captured["token_ids"])
    assert decode_seq.status is SequenceStatus.FINISHED
    assert plan[1].seq.status is SequenceStatus.RUNNING
    assert plan[1].seq.num_completion_tokens == 1
    drain(llm, SAMPLE_SEED + 12)
    assert_no_owned_kv_blocks(llm)
    print("PASS: mixed [decode, complete-prefill] samples both requests")


def run_partial_prefill_mixed_case(llm: LLM) -> None:
    """A partial prompt commits KV only; it must not consume sampling randomness."""
    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    decode_seq = make_running(llm, token_ids(500, 6), params, SAMPLE_SEED + 20)
    llm.add_request(token_ids(600, 20), params)

    plan, captured, _ = run_engine_step(llm, SAMPLE_SEED + 21, observe_tensors=True)
    assert [item.is_prefill for item in plan] == [False, True]
    decode_item, prefill_item = plan
    assert decode_item.seq is decode_seq
    assert prefill_item.scheduled_tokens == MAX_BATCH_TOKENS - 1
    assert prefill_item.cached_tokens == 0
    assert sum(item.scheduled_tokens for item in plan) == MAX_BATCH_TOKENS
    assert len({item.seq.seq_id for item in plan}) == len(plan)

    assert captured["run_model_calls"] == 1
    assert captured["use_varlen"] is True
    assert captured["input_ids"] == [decode_item.last_token, *token_ids(600, prefill_item.scheduled_tokens)]
    assert captured["positions"] == [decode_item.cached_tokens, *range(prefill_item.scheduled_tokens)]
    assert captured["cu_seqlens_q"] == [0, 1, MAX_BATCH_TOKENS]
    assert captured["cu_seqlens_k"] == [0, decode_item.cached_tokens + 1, decode_item.cached_tokens + 1 + prefill_item.scheduled_tokens]
    assert captured["slot_mapping_size"] == MAX_BATCH_TOKENS
    assert captured["block_table_shape"][0] == 2

    assert isinstance(captured["token_ids"][0], int)
    assert captured["token_ids"][1] is None
    assert decode_seq.status is SequenceStatus.FINISHED
    partial_seq = prefill_item.seq
    assert partial_seq.status is SequenceStatus.WAITING
    assert partial_seq.num_cached_tokens == prefill_item.scheduled_tokens
    assert partial_seq.num_completion_tokens == 0
    assert list(llm.scheduler.waiting) == [partial_seq]
    drain(llm, SAMPLE_SEED + 22)
    assert_no_owned_kv_blocks(llm)
    print("PASS: mixed [decode, partial-prefill] respects budget and skips partial sampling")


def run_partial_prefill_rng_case(llm: LLM) -> None:
    """Partial prefill must leave sampling RNG exactly as a one-row decode would."""
    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    prompt = token_ids(900, 6)

    make_running(llm, prompt, params, SAMPLE_SEED + 25)
    _, reference, _ = run_engine_step(llm, SAMPLE_SEED + 26)
    reference_state = torch.cuda.get_rng_state()
    reference_token = reference["token_ids"][0]
    assert_no_owned_kv_blocks(llm)

    make_running(llm, prompt, params, SAMPLE_SEED + 25)
    llm.add_request(token_ids(10000, 20), params)
    plan, mixed, _ = run_engine_step(llm, SAMPLE_SEED + 26)
    mixed_state = torch.cuda.get_rng_state()
    assert [item.is_prefill for item in plan] == [False, True]
    assert mixed["token_ids"] == [reference_token, None]
    assert torch.equal(mixed_state, reference_state)
    drain(llm, SAMPLE_SEED + 27)
    assert_no_owned_kv_blocks(llm)
    print("PASS: partial prefill leaves CUDA sampling RNG identical to one decode")


def run_sequence_limit_case(llm: LLM) -> None:
    """Existing decode entries keep FIFO order and consume all sequence slots first."""
    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    for start in (700, 720, 740, 760):
        llm.add_request(token_ids(start, 3), params)
    plan, _, _ = run_engine_step(llm, SAMPLE_SEED + 30)
    assert [item.is_prefill for item in plan] == [True] * 4
    running = [item.seq for item in plan]
    assert all(seq.status is SequenceStatus.RUNNING for seq in running)

    llm.add_request(token_ids(800, 5), params)
    plan, _, _ = run_engine_step(llm, SAMPLE_SEED + 31)
    assert [item.is_prefill for item in plan] == [False] * 4
    assert [item.seq for item in plan] == running
    assert sum(item.scheduled_tokens for item in plan) == 4
    assert len(plan) == llm.scheduler.max_num_seqs
    assert len(llm.scheduler.waiting) == 1
    assert all(seq.status is SequenceStatus.FINISHED for seq in running)

    plan, _, _ = run_engine_step(llm, SAMPLE_SEED + 32)
    assert [item.is_prefill for item in plan] == [True]
    drain(llm, SAMPLE_SEED + 33)
    assert_no_owned_kv_blocks(llm)
    print("PASS: sequence limit preserves decode-first order and leaves new prefill waiting")


def run_prefix_cache_mixed_case(llm: LLM) -> None:
    """A cached 256-token prefix contributes computed KV without consuming budget."""
    source_params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    prefix = token_ids(1000, 256)
    source_prompt = prefix + [1256]
    llm.add_request(source_prompt, source_params)
    drain(llm, SAMPLE_SEED + 40)
    assert llm.scheduler.block_manager.hash_to_block_id

    params = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)
    decode_seq = make_running(llm, token_ids(1300, 6), params, SAMPLE_SEED + 41)
    llm.add_request(prefix + [2256], params)
    plan, captured, _ = run_engine_step(llm, SAMPLE_SEED + 42, observe_tensors=True)
    assert [item.is_prefill for item in plan] == [False, True]
    assert plan[0].seq is decode_seq
    cached_prefill = plan[1]
    assert cached_prefill.cached_tokens == 256
    assert cached_prefill.scheduled_tokens == 1
    assert sum(item.scheduled_tokens for item in plan) == 2
    assert captured["use_varlen"] is True
    assert captured["block_table_shape"][0] == 2
    assert isinstance(captured["token_ids"][1], int)
    assert cached_prefill.seq.status is SequenceStatus.RUNNING
    drain(llm, SAMPLE_SEED + 43)
    assert_no_owned_kv_blocks(llm)
    print("PASS: prefix-cache hit joins a decode without charging cached tokens to the budget")


def run_preemption_control_plane_case() -> None:
    """Exercise scheduler preemption with deterministic, intentionally tiny KV capacity.

    Real GPUs expose a hardware-dependent number of KV blocks, so exhaustion is
    not reproducible across hosts. This fixture checks the scheduler-only
    ownership transition while all model-execution cases above remain real GPU
    engine steps.
    """
    config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=1,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=2,
    )
    scheduler = Scheduler(config)
    seq = Sequence(token_ids(3000, 257), SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True))
    scheduler.block_manager.allocate(seq, num_cached_blocks=0)
    seq.num_cached_tokens = 256
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)

    output = scheduler.schedule()
    assert [item.is_prefill for item in output.scheduled] == [True]
    assert output.seqs == [seq]
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_scheduled_tokens == 1
    assert seq.block_table
    assert list(scheduler.waiting) == [seq]
    print("PASS: KV exhaustion preempts decode, releases ownership, and requeues it as prefill")


def run_eos_control_plane_case() -> None:
    """Check EOS completion and KV release without depending on model output."""
    config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=1,
        eos=42,
        kvcache_block_size=256,
        num_kvcache_blocks=1,
    )
    scheduler = Scheduler(config)
    seq = Sequence([1], SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=False))
    scheduler.block_manager.allocate(seq, num_cached_blocks=0)
    seq.num_scheduled_tokens = 1
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)

    scheduler.postprocess(SchedulerOutput([ScheduledSequence(seq, is_prefill=False)]), [42])
    assert seq.status is SequenceStatus.FINISHED
    assert not seq.block_table
    assert not scheduler.running
    assert not scheduler.block_manager.used_block_ids
    print("PASS: EOS completion releases KV ownership and removes the running request")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local Qwen3-0.6B model directory")
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    assert torch.cuda.is_available(), "This test requires a CUDA GPU"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    llm = LLM(
        str(args.model),
        max_num_batched_tokens=MAX_BATCH_TOKENS,
        max_num_seqs=4,
        max_model_len=512,
        enforce_eager=False,
    )
    run_pure_path_regression(llm)
    run_complete_prefill_mixed_case(llm)
    run_partial_prefill_mixed_case(llm)
    run_partial_prefill_rng_case(llm)
    run_sequence_limit_case(llm)
    run_prefix_cache_mixed_case(llm)
    run_preemption_control_plane_case()
    run_eos_control_plane_case()
    print("M5/M6 mixed-batch GPU correctness suite: PASS")


if __name__ == "__main__":
    main()
