import gc
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

QUESTIONS = [
    "Provide summary of the book.",
    "What does Gregor Samsa discover when he wakes up at the beginning of the story?",
    "How does Gregor Samsa die and what happens to his family after his death?",
    "How does Gregor's sister Grete change her attitude toward him throughout the story?",
    "How does the family's financial situation change from the beginning to the end of the story?",
]

LOGPROB_ATOL = 0.05
BOOK_PATH = Path("./metamorphosis_franz_kafka.txt")


def get_ctx(file_path: Path) -> str:
    with open(file_path, "r") as file:
        return file.read()


def make_llm(impl_type: str) -> LLM:
    return LLM(
        model="../models/gemma3-1b-it",
        tensor_parallel_size=1,
        max_model_len=32768,
        block_size=64,
        enable_prefix_caching=True,
        model_impl=impl_type,
    )


def get_logprobs(llm: LLM, context: str, question: str, params: SamplingParams) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions based only on "
                "the provided book text. Be concise."
            ),
        },
        {
            "role": "user",
            "content": f"Here is the full text of a book:\n\n{context}\n\nQuestion: {question}",
        },
    ]
    outputs = llm.chat(messages, params)
    raw = outputs[0].outputs[0].logprobs
    return {
        pos: {str(tid): (lp.decoded_token, lp.logprob) for tid, lp in token_lps.items()}
        for pos, token_lps in enumerate(raw)
    }


def run_impl(impl_type: str, context: str, params: SamplingParams) -> dict:
    print(f"\n{'═' * 60}")
    print(f"  Running: {impl_type}")
    print(f"{'═' * 60}")

    llm = make_llm(impl_type)
    results = {}

    for i, question in enumerate(QUESTIONS, 1):
        print(f"  [{i}/{len(QUESTIONS)}] {question[:60]}...")
        t0 = time.time()
        results[question] = get_logprobs(llm, context, question, params)
        print(f"  done in {time.time() - t0:.2f}s, {len(results[question])} tokens")

    del llm
    gc.collect()

    return results


def compare(ref_impl: str, ref: dict, cur_impl: str, cur: dict):
    print(f"\n{'═' * 60}")
    print(f"  Comparing: {ref_impl} (ref)  vs  {cur_impl}")
    print(f"{'═' * 60}")

    total_positions = 0
    total_mismatches = 0
    total_diffs = []

    for i, question in enumerate(QUESTIONS, 1):
        ref_lps = ref[question]
        cur_lps = cur[question]
        min_len = min(len(ref_lps), len(cur_lps))
        mismatches = []

        for pos in range(min_len):
            ref_pos = ref_lps[pos]
            cur_pos = cur_lps[pos]

            ref_top_id = max(ref_pos, key=lambda tid: ref_pos[tid][1])
            cur_top_id = max(cur_pos, key=lambda tid: cur_pos[tid][1])

            ref_token, ref_lp = ref_pos[ref_top_id]
            cur_token, cur_lp = cur_pos[cur_top_id]

            diff = abs(ref_lp - cur_lp)
            total_diffs.append(diff)
            total_positions += 1

            if ref_top_id != cur_top_id or diff > LOGPROB_ATOL:
                mismatches.append((pos, ref_token, ref_lp, cur_token, cur_lp, diff))
                total_mismatches += 1

        print(f"\n  [{i}/{len(QUESTIONS)}] {question[:60]}...")
        if mismatches:
            print(f"  ❌ {len(mismatches)}/{min_len} mismatches")
            for pos, rt, rl, ct, cl, d in mismatches[:3]:
                print(f"     pos {pos:3d}: {ref_impl}={rt!r:15s}({rl:.4f})  "
                      f"{cur_impl}={ct!r:15s}({cl:.4f})  diff={d:.4f}")
            if len(mismatches) > 3:
                print(f"     ... and {len(mismatches) - 3} more")
        else:
            print(f"  ✅ all {min_len} positions match")

    avg_diff = sum(total_diffs) / len(total_diffs) if total_diffs else 0
    max_diff = max(total_diffs) if total_diffs else 0

    print(f"\n{'═' * 60}")
    print(f"  total positions : {total_positions}")
    print(f"  mismatches      : {total_mismatches} ({100 * total_mismatches / max(total_positions, 1):.1f}%)")
    print(f"  avg logprob diff: {avg_diff:.4f}")
    print(f"  max logprob diff: {max_diff:.4f}")
    print(f"  {'✅ PASS' if total_mismatches == 0 else '❌ FAIL'}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    context = get_ctx(BOOK_PATH)
    print(f"Loaded {len(context):,} characters from {BOOK_PATH}")

    params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        logprobs=10,
    )

    start = time.time()

    ref_results = run_impl("vllm", context, params)
    print(ref_results)
    cur_results = run_impl("flax_nnx", context, params)
    print(cur_results)
    compare("vllm", ref_results, "flax_nnx", cur_results)

    print(f"Total time: {time.time() - start:.2f}s")