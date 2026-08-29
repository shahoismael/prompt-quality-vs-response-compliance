# Prompt Quality vs Response Compliance

Code, data and results for a study measuring prompt quality and response compliance as separate constructs, using a locally hosted open-weights LLM judge on consumer hardware.

**Headline result.** Independently measured prompt quality does not predict response compliance when compliance is scored against prompt-independent criteria (ρ = 0.014, 90% CI [−0.062, 0.090], TOST *p* = .031, n = 498). Response quality is not a proxy for prompting skill.

The manuscript is not in this repository.

---

## What is here

```
src/          MATLAB pipeline (agents, runner, baselines, analysis)
scripts/      IFEval verification (Python) and figure generation
data/         IFEval items and a hash manifest of the full benchmark
results/      Per-item numeric records (CSV) and verification output
```

**No prompt or response text from WildChat-1M or LMSYS-Chat-1M is redistributed here.** The LMSYS-Chat-1M agreement prohibits transferring the dataset to third parties, and WildChat-1M carries flow-down obligations under the AI2 ImpACT licence. Rebuilding the corpus is a two-step process described under [Data](#data) below.

## Requirements

| | |
|---|---|
| MATLAB | R2025b (base only, no toolboxes) |
| Python | 3.10+, for `scripts/` |
| Ollama | serving `qwen2.5:7b` at `127.0.0.1:11434` |
| Hardware | CPU is sufficient; the reported run used an i7-1255U with 40 GB RAM |

```bash
pip install h5py numpy matplotlib
pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval
```

`Ollama_API.m` targets `127.0.0.1` rather than `localhost`. On Windows, `localhost` resolves to the IPv6 loopback, where Ollama does not listen.

## Reproducing

Run everything from the repository root, not from `src/`. Paths are relative to the working directory.

```matlab
cd <repo root>
addpath('src'); addpath('scripts')

run_simulation(500, 'results/simulation_results_main500_v2.mat')   % ~58 h CPU
run_baseline('results/simulation_results_main500_v2.mat', ...
             'results/baseline_results_v2.mat', 120)
validate_construct
analyze_ablation('results/simulation_results_main500_v2.mat', ...
                 'results/baseline_results_v2.mat')
make_figs_data
```

```bash
python scripts/verify_ifeval.py            # official IFEval verification
python scripts/make_figs_concept.py <out>  # conceptual figures
```

A single seed (`rng(42,'twister')`) governs item sampling, perturbation and known-groups degradation, so the item set is exactly reproducible. Model sampling is deliberately **unseeded**: a pinned seed makes ensemble members identical, drives sample variance to zero, and collapses both the confidence estimate and the stability metric. Scores therefore reproduce in distribution, not byte for byte.

## Architecture

| Component | File | Role |
|---|---|---|
| Extractor | `src/ExtractorAgent.m` | Parses a prompt into strict and loose criteria |
| Prompt Quality | `src/PromptQualityAgent.m` | Scores the prompt before any response exists |
| Semantic Scorer | `src/SemanticScorer.m` | Scores the response against extracted criteria |
| Calibration | `src/CalibrationAgent.m` | Removes the length-attributable score component |
| Fixed rubric | `src/FixedRubricScorer.m` | Prompt-independent control arm |

`FixedRubricScorer` never sees the prompt. The contrast between it and the extracted-criteria arm, on identical responses, is what the criterion-source analysis rests on.

## Two things worth knowing before reusing this code

**Screen model output before fitting any length correction.** Two of 500 responses were repetition loops of 24,271 and 33,864 words. Fitted with them included, the length regression returns β₁ = −3.8 × 10⁻⁷ with R² = 2.5 × 10⁻⁷, and reports no verbosity bias. Screened out (type-token ratio ≥ 0.10), the same regression returns β₁ = +1.57 × 10⁻³, *t* = 13.0, R² = 0.256. Least squares places no bound on the leverage of a single observation, and generative pipelines produce such observations routinely.

**Never treat an unverifiable constraint as a failed one.** The in-pipeline rule-based checker evaluates 21.6% of the strict constraints proposed for IFEval items. Scoring accuracy as the proportion of stored values equal to 1 turns each coverage gap into a failure and reports 0.289 where the verified figure is 0.681. `scripts/verify_ifeval.py` computes the definitive figures against the benchmark's own `instructions_registry`.

## Data

| File | Contents |
|---|---|
| `data/ifeval_filtered.json` | IFEval items with `instruction_id_list` and `kwargs` retained (Apache 2.0) |
| `data/benchmark_manifest.csv` | All 4,275 benchmark rows as `idx, source, prompt_sha256, prompt_words` |

The manifest carries no prompt text. It lets you confirm that a corpus you rebuilt yourself is byte-identical to the one used here: hash each prompt with SHA-256 and compare.

To rebuild the corpus:

1. Obtain [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) and [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) directly from their maintainers, accepting each licence yourself.
2. Run `src/create_master_benchmark.m`, which applies the filters described in the paper's Section 3.2 and reproduces `master_benchmark.json` in the original row order.
3. Verify against `data/benchmark_manifest.csv`.

One IFEval item (index 339) had a constraint value that diverged from the reference release and is corrected here; it was excluded from the reported accuracy figures rather than spliced back in.

Both naturalistic corpora retain hashed IP addresses and coarse geographic metadata upstream. That is pseudonymisation, not full de-identification, which is a second reason the text is not republished here.

## Results files

| File | Contents |
|---|---|
| `main_run_500.csv` | Main run, one row per item: prompt hash, both prompt-quality composites, all three compliance measures, bias variance, routing counts, latency, screen flag |
| `baseline_120.csv` | Naive and chain-of-thought baselines on the aligned subsample, keyed to `main_run_idx` |
| `construct_validation.mat` | Known-groups validation, 100 paired prompts, numeric only |
| `ifeval_verified.json` | Official verification output, per-item and per-instruction-type |
| `ifeval_responses.jsonl` | Generated responses for the IFEval subset (IFEval is Apache 2.0) |

Every reported statistic in the paper is recomputable from these two CSVs. `S_cal` is the length-corrected score after the response screen; `kept = 0` marks the two items the screen removes.

The full `.mat` run files contain the source prompts and are therefore withheld for the licence reasons above. They are available from the authors for verification, subject to the requester holding their own dataset agreements.

## Licence

Code released under the MIT Licence (`LICENSE`). Data files are governed by their upstream licences, listed above.
