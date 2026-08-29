#!/usr/bin/env python3
"""
Definitive IFEval strict/loose accuracy, using Google's OFFICIAL verifier.

WHY THIS FILE EXISTS
--------------------
run_simulation.m previously computed "strict accuracy" like this:

  * the Extractor LLM read the prompt and invented free-text constraint
    descriptions;
  * check_strict_constraint() tried to match those descriptions against nine
    hardcoded keyword patterns ('bullet', 'word count', 'uppercase', ...);
  * anything it could not match was stored as NaN;
  * strict accuracy was mean(scores == 1), and because NaN == 1 is false,
    every UNVERIFIABLE constraint was silently counted as a FAILURE.

IFEval defines 25 instruction types. Nine keyword patterns cannot cover them,
so most constraints fell through to NaN and were scored zero. That is the
likely origin of the 6.4% strict-accuracy figure.

Worse, data/master_benchmark.json stores each IFEval item's real
instruction_id_list and run_simulation.m never read it. The benchmark's own
ground truth was loaded and then ignored, and strict accuracy was measured
against LLM-generated labels -- exactly the ground-truth circularity the
manuscript criticises in other work.

This script fixes that. It scores the saved responses with the canonical
implementation, using each item's real instruction ids and kwargs, and
reports strict and loose accuracy under the official definitions
(prompt-level = all instructions satisfied; instruction-level = fraction of
instructions satisfied; loose applies the official response transformations).

SETUP (once, on a machine with internet)
----------------------------------------
    pip install absl-py langdetect nltk immutabledict

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/google-research/google-research.git
    cd google-research
    git sparse-checkout set instruction_following_eval

Then either copy the `instruction_following_eval` folder next to this script,
or pass its parent directory with --registry.

The clone also provides instruction_following_eval/data/input_data.jsonl,
which carries the kwargs (num_words, relation, num_highlights, ...) that
data/ifeval_filtered.json does not store. This script joins your saved
responses to it on exact prompt text.

USAGE
-----
    python verify_ifeval.py \
        --responses results/ifeval_responses.jsonl \
        --input-data google-research/instruction_following_eval/data/input_data.jsonl \
        --registry   google-research \
        --out        results/ifeval_verified.json
"""

import argparse
import json
import os
import sys
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalise(text):
    """Join key for matching a saved response back to its benchmark record."""
    return " ".join((text or "").split())


# ---- official loose-accuracy response transformations (IFEval paper, §3) ----
def loose_variants(response):
    r = response
    stripped = r.strip()

    no_first_line = "\n".join(r.split("\n")[1:]).strip()
    no_last_line = "\n".join(r.split("\n")[:-1]).strip()

    def demark(t):
        return t.replace("*", "")

    variants = [
        stripped,
        demark(stripped),
        no_first_line,
        no_last_line,
        demark(no_first_line),
        demark(no_last_line),
    ]
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_instruction(registry, instruction_id, kwargs, prompt):
    cls = registry.INSTRUCTION_DICT[instruction_id]
    inst = cls(instruction_id)
    inst.build_description(**(kwargs or {}))
    args = inst.get_instruction_args()
    if args and "prompt" in args:
        inst.build_description(prompt=prompt)
    return inst


def score_item(registry, record, response):
    """Return (strict_all, loose_all, per_instruction_strict, n_instructions)."""
    prompt = record["prompt"]
    ids = record.get("instruction_id_list", []) or []
    kwlist = record.get("kwargs", [{}] * len(ids)) or [{}] * len(ids)

    strict_flags, loose_flags = [], []
    for iid, kw in zip(ids, kwlist):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        try:
            inst = build_instruction(registry, iid, kw, prompt)
        except Exception as exc:  # unknown id / bad kwargs -> report, never pass
            print(f"  ! could not build {iid}: {exc}", file=sys.stderr)
            strict_flags.append(False)
            loose_flags.append(False)
            continue

        try:
            ok_strict = bool(response.strip()) and inst.check_following(response)
        except Exception:
            ok_strict = False
        strict_flags.append(ok_strict)

        ok_loose = False
        for variant in loose_variants(response):
            try:
                if inst.check_following(variant):
                    ok_loose = True
                    break
            except Exception:
                continue
        loose_flags.append(ok_loose)

    return strict_flags, loose_flags, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="results/ifeval_responses.jsonl")
    ap.add_argument("--input-data", required=True,
                    help="instruction_following_eval/data/input_data.jsonl from the official repo")
    ap.add_argument("--registry", default=".",
                    help="directory CONTAINING the instruction_following_eval package")
    ap.add_argument("--out", default="results/ifeval_verified.json")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.registry))
    try:
        from instruction_following_eval import instructions_registry as registry
    except ImportError as exc:
        sys.exit(
            f"Could not import the official verifier ({exc}).\n"
            "Run the SETUP block at the top of this file, then pass --registry\n"
            "pointing at the directory that CONTAINS instruction_following_eval/."
        )

    bench = {normalise(r["prompt"]): r for r in load_jsonl(args.input_data)}
    responses = load_jsonl(args.responses)
    print(f"benchmark records: {len(bench)} | saved responses: {len(responses)}")

    matched, unmatched = [], []
    for row in responses:
        key = normalise(row["prompt"])
        if key in bench:
            matched.append((bench[key], row["response"]))
        else:
            unmatched.append(row["prompt"][:80])

    if unmatched:
        print(f"WARNING: {len(unmatched)} responses did not match a benchmark prompt.")
        for p in unmatched[:5]:
            print("   unmatched:", p)
        print("   -> the prompts in master_benchmark.json were altered relative to")
        print("      the official IFEval input_data.jsonl. Fix before reporting.")

    n_prompt_strict = n_prompt_loose = 0
    inst_strict = inst_loose = inst_total = 0
    per_type = Counter()
    per_type_ok = Counter()
    detail = []

    for record, response in matched:
        s_flags, l_flags, n_inst = score_item(registry, record, response)
        if n_inst == 0:
            continue
        n_prompt_strict += int(all(s_flags))
        n_prompt_loose += int(all(l_flags))
        inst_strict += sum(s_flags)
        inst_loose += sum(l_flags)
        inst_total += n_inst
        for iid, ok in zip(record["instruction_id_list"], s_flags):
            per_type[iid] += 1
            per_type_ok[iid] += int(ok)
        detail.append({
            "key": record.get("key"),
            "instruction_id_list": record["instruction_id_list"],
            "strict": s_flags,
            "loose": l_flags,
        })

    n = len(detail)
    if n == 0:
        sys.exit("No scorable items. Check --responses and --input-data.")

    summary = {
        "n_items": n,
        "n_instructions": inst_total,
        "prompt_level_strict_accuracy": n_prompt_strict / n,
        "prompt_level_loose_accuracy": n_prompt_loose / n,
        "instruction_level_strict_accuracy": inst_strict / inst_total,
        "instruction_level_loose_accuracy": inst_loose / inst_total,
        "unmatched_responses": len(unmatched),
        "per_instruction_type": {
            k: {"n": per_type[k], "passed": per_type_ok[k], "rate": per_type_ok[k] / per_type[k]}
            for k in sorted(per_type)
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "detail": detail}, fh, indent=2)

    print("\n===== OFFICIAL IFEVAL VERIFICATION =====")
    print(f"items scored              : {n}")
    print(f"instructions scored       : {inst_total}")
    print(f"prompt-level   STRICT acc : {summary['prompt_level_strict_accuracy']:.4f}")
    print(f"prompt-level   LOOSE  acc : {summary['prompt_level_loose_accuracy']:.4f}")
    print(f"instr-level    STRICT acc : {summary['instruction_level_strict_accuracy']:.4f}")
    print(f"instr-level    LOOSE  acc : {summary['instruction_level_loose_accuracy']:.4f}")
    print("\nweakest instruction types:")
    worst = sorted(per_type, key=lambda k: per_type_ok[k] / per_type[k])[:8]
    for k in worst:
        print(f"  {k:45s} {per_type_ok[k]:4d}/{per_type[k]:<4d}  {per_type_ok[k]/per_type[k]:.2f}")
    print(f"\nwritten to {args.out}")
    print("Report the instruction-level figures as the paper's Strict/Loose Accuracy.")
    print("========================================")


if __name__ == "__main__":
    main()
