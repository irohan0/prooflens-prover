# ProofLens-Prover — retrieval architecture in a live Lean 4 theorem prover

A controlled study of **premise retrieval inside a working Lean 4 proof search**: does the retrieval
architecture change how many theorems a prover can close, and if so, where and why?

Every arm shares one search harness, one premise corpus, one Lean toolchain and one search budget.
Only the retriever varies. Every claimed proof is re-elaborated from the benchmark statement in a
fresh Lean environment before it is counted.

---

## What this system is — and what it is not

**There is no language model in this prover.** The tactic policy is a fixed repertoire of 19 Lean
tactics (`simp`, `aesop`, `linarith`, `intro x`, …) plus five templates that consume a retrieved
premise (`exact {p}`, `apply {p}`, `rw [{p}]`, `rw [← {p}]`, `simp [{p}]`). Every number below comes
from that policy.

This is a deliberate choice, not an unfinished one. With no generator there is no prompt sensitivity,
no sampling variance and no training confound, so **any difference between arms is attributable to
the retriever alone**. The price is that absolute pass rates are a floor rather than a competitive
result, and are not comparable to systems built on fine-tuned 7B models.

The generator is the next stage and has not started:

| stage | generator | status |
|---|---|---|
| **now** | model-free repertoire (19 tactics + 5 premise templates) | ✅ complete, results below |
| **next** | REAL-Prover-v1 — 7B, `Qwen2.5-Math-7B` base, stepwise, retrieval-native | not started |
| later | Qwen3-4B / Kimina-Prover-Distill-1.7B, LoRA SFT | planned |

REAL-Prover-v1 is chosen because it is **openly released and already retrieval-native** — it was
trained to consume premises from its own single-vector retriever, `LeanSearch-PS`. That lets the
generator be frozen and only the retriever swapped, which is the same controlled design used here,
at a scale where the numbers are comparable to published work (their FATE-M 56.7 with retrieval,
44.7 without).

`Qwen2.5-Math-1.5B` was ruled out on a hard constraint rather than a preference: a **4K context
window** cannot hold retrieved premises alongside a proof state.

---

## Results

Retrieval arms over the model-free policy described above.

| benchmark | none (floor) | ProofLens-SV | ProofLens-LI | Δ LI vs none | Δ LI vs SV |
|---|--:|--:|--:|--:|--:|
| **FATE-M** (141) | 12 (8.5%) | **35 (24.8%)** | 22 (15.6%) | **+10** (p=0.0020) | **−13** (p=0.0023) |
| ProofNet-test (186) | 9 (4.8%) | — | 12 (6.5%) | +3 (p=0.25) | — |
| miniF2F-test (244) | 78 (32.0%) | — | 78 (32.0%) | 0 (p=1.00) | — |

Significance requires the paired bootstrap CI (10k) and the sign-flip permutation test (10k) to
**agree**; the exact McNemar p is reported since the outcome is binary and paired.

### 1. Retrieval helps, and only where premises are needed

Against the no-retrieval control, LI gains **+10 problems on FATE-M** (95% CI [+0.028, +0.114],
permutation p = 0.0023). The control's solved set is a strict subset of LI's — retrieval never cost
a problem.

The effect tracks whether the benchmark *requires* citing lemmas:

| benchmark | domain | Δ vs none |
|---|---|--:|
| FATE-M | graduate abstract algebra | **+10** |
| ProofNet-test | undergraduate, mixed | +3 |
| miniF2F-test | competition arithmetic | **0** |

miniF2F is the strongest form of a null: not merely equal counts but **identical solved sets**. Its
problems fall to `linarith` / `nlinarith` / `omega` / `norm_num`, where retrieval has no leverage.

### 2. "Premise used" is not "premise needed"

Two distinct quantities, routinely conflated:

* **used** — of all proofs found, how many name a retrieved premise. FATE-M: **16/22 = 73%**.
* **needed** — of the proofs the *control could not find*, how many name a premise. FATE-M:
  **10/10 = 100%**.

Only the second is causal. Six FATE-M proofs used a premise (`exact conj_pow`, `rw [orderOf_inv]`,
`apply Fintype.card_pi_const`, …) for problems the control also solved with a bare `simp`/`aesop` —
the premise tactic merely ranked first. On miniF2F the gap is total: 22% used, 0% needed.

Distinguishing them requires a paired control. Reading proof text cannot do it.

### 3. Single-vector beat multi-vector — with a confound that is not yet excluded

On FATE-M, SV solved **35/141** against LI's **22/141**: −13 problems, p = 0.0023, with all 35 and
all 22 proofs independently verified. This is the opposite of the study's hypothesis and it is
reported as measured. There is also real displacement — 15 problems only SV solved, 2 only LI
solved — so the arms are not nested.

**However, the arms are not computationally matched, and the asymmetry favours SV.** The cause is
the defining property of late interaction: it keeps **one vector per token** instead of one per
premise.

| | vectors stored | index size | exact search |
|---|--:|--:|---|
| ProofLens-SV | 276,070 (one per premise) | 943 MB | one matrix-vector product — **feasible** |
| ProofLens-LI | **21,752,080** (78.8 per premise) | 5.5 GB fp16 | see below — **infeasible** |

Exact full-corpus MaxSim requires scoring every query token against every premise token: a
`384 × 21,752,080` score matrix, ≈ **8.4 billion floats (33 GB) for a single query**. At roughly
5,300 queries per benchmark run that is not an engineering inconvenience, it is a different
algorithm.

So LI is necessarily **two-stage**: mean-pooled vectors select the top 1,000 candidates — **0.36% of
the corpus** — and MaxSim only ever sees those. A premise the first stage drops cannot be recovered
by late interaction, no matter how good the late interaction is. SV has no such stage; it ranks all
276,070 exactly.

The `recall@10 = 0.992` printed at index-build time does **not** license ignoring this. It was
measured with *premise embeddings* as probes, so it describes a premise retrieving its neighbours —
not a proof state retrieving a premise. Wrong distribution, optimistic by an unknown margin.

**So the honest reading today is "exact single-vector beats approximate multi-vector at a 0.36%
first-stage budget."** That is a claim about candidate generation, not about late interaction, and
the two have very different implications. `scripts/measure_li_recall.py` separates them; see
[H1 in Next steps](#next-steps-as-hypotheses).

### Cost

On FATE-M, SV is both more accurate **and about twice as fast per query** — the practical case for
late interaction here is weak on both axes at this first-stage budget.

| | size | ms/query |
|---|--:|--:|
| BM25 | 169 MB | **9.8** |
| ProofLens-SV (768-dim) | 943 MB | **44.7** |
| ProofLens-LI (fp16, 21.7M token vectors, 78.8/premise) | ~5.5 GB | **85.9** |

| | |
|---|--:|
| Premise corpus | 276,070 premises (Mathlib v4.16.0) |
| `import Mathlib`, NFS | 439–691 s |
| `import Mathlib`, node-local | **158 s** |

LI at `query_length=256` measured 77.9 ms; raising it to the locked 384 cost ~10% latency and
changed which premises were retrieved (5,288 → 5,330 retrieval calls) but left the solved set
unchanged — the problems a model-free policy closes have short goals, where truncation never bit.

### Verification

`scripts/verify_proofs.py` re-elaborates every claimed proof from the benchmark statement in a fresh
Lean environment, sharing nothing with the search but the cheat-token regex. `sorry`, `admit` and
`native_decide` are rejected before execution and again at verification.

| run | verified |
|---|---|
| FATE-M / SV | **35/35 ✅** |
| FATE-M / LI | 22/22 ✅ |
| FATE-M / none | ✅ |
| ProofNet / LI | ✅ |
| miniF2F / LI | 78/78 ✅ |

Every reported proof re-elaborates cleanly. The results table is not provisional.

### Run records

`results/exported/` holds each run's manifest and per-problem outcomes (traces stripped; they are
the bulk of the file and no reported number depends on them). The table above regenerates from them:

```bash
python scripts/build_table1.py --results-root results/exported/logs
```

Superseded runs are kept rather than deleted, because two of them are the evidence for defects
described in the design notes. `build_table1.py` selects the most recent *finalised* run per
(benchmark, arm), so their presence does not affect the table.

| run | note |
|---|---|
| `..._5fcb0b4` | 5-problem smoke run |
| `..._027cff0` | superseded: a REPL restart voided problems 109–140 |
| `..._a59423a` | superseded: `query_length=256` |
| `..._9cc3513` | current: `query_length=384` |

---

## Conclusions

1. **Retrieval measurably improves a live Lean prover**, but only on benchmarks whose proofs require
   citing lemmas the policy cannot otherwise reach. On competition arithmetic the effect is exactly
   zero. Benchmark choice determines whether a retrieval claim is measurable at all.
2. **Retrieval attribution needs a paired control.** The intuitive metric — how many proofs cite a
   retrieved premise — overstated the causal contribution by 60% on FATE-M.
3. **Multi-vector retrieval did not beat single-vector here**, under a first-stage budget of 0.36%
   of the corpus. Whether that is a fact about late interaction or about approximate candidate
   generation is the open question, and the experiment that separates them is specified below.
4. **A model-free policy reaches 8.5–32% depending on benchmark**, which sets the floor any
   generator must clear to demonstrate it is contributing.

## Limitations

* The policy is model-free; absolute rates are **not** comparable to 7B-class systems
  (REAL-Prover-v1: 56.7 FATE-M / 23.7 ProofNet; ReProver: 13.8 ProofNet).
* ProofNet's +3 cannot reach significance: with zero discordant pairs on the other side, McNemar
  needs ≥ 6. It is underpowered by construction, not weakly positive.
* 4 ProofNet statements do not elaborate under Lean v4.16.0; the effective denominator is 182. 186
  is reported for comparability with published numbers.
* `AutoLeanServer` restarts the REPL on system-wide memory pressure, which on a shared cluster can
  be triggered by other jobs. Recoveries are counted in each manifest (`n_stale_env_recoveries`).
* Retrieval queries the *statement* at the root and the pretty-printed goal thereafter; no proof
  context beyond the goal is used.

---

## Next steps, as hypotheses

Each step is stated as a claim that the experiment can falsify, with what each outcome would mean.

### H1 — LI lost to its candidate generator, not to late interaction

> **Claim.** LI's two-stage retrieval drops relevant premises before MaxSim ever sees them. At a
> first-stage budget large enough to recover them, LI's FATE-M score rises materially above 22/141.

**Test.** `scripts/measure_li_recall.py` computes exact full-corpus MaxSim in chunks for 40 real
queries and reports recall@10 at n_candidates = 1,000 / 5,000 / 20,000. Then re-run the LI arm at
whichever budget the recall curve saturates.

| outcome | reading |
|---|---|
| recall@1,000 ≈ 0.99 | the approximation is not the problem; **SV genuinely beats LI here**, and the finding stands as an architecture result |
| recall@1,000 ≪ 0.99, LI improves when re-run | the published −13 measured **candidate generation**, not late interaction; the arm comparison must be redone at matched recall |
| recall low but LI does **not** improve | late interaction is not exploiting the premises it recovers — the more interesting negative result, and one worth a section of its own |

**This is the highest-value job remaining.** It decides which of two very different theses the
project has produced, and it costs one GPU job.

### H2 — the benchmark, not the retriever, determines whether retrieval can matter

> **Claim.** Retrieval's measurable effect is a function of how much a benchmark's proofs depend on
> citing lemmas, and is near zero where tactic automation suffices.

Already supported: FATE-M +10, ProofNet +3, miniF2F exactly 0 with **identical solved sets**.
Strengthening it needs the SV arm on ProofNet and miniF2F, and the LI re-runs at
`query_length=384`. Lower priority: on FATE-M the query-length change altered retrieval without
altering the outcome, so the expected effect is small.

### H3 — the generator, not the search budget, is the binding constraint

> **Claim.** The model-free policy is limited by the tactics it can propose, not by how long it may
> search. Adding a language model moves the numbers; adding search budget does not.

Already supported: **51 of 71** exhausted FATE-M searches hit `max_expansions`, not the wall clock.
The search runs out of ideas, not time.

**Test.** Tier 1 — freeze REAL-Prover-v1 (7B) and swap only the retriever, reproducing their
published FATE-M 56.7 with `LeanSearch-PS` as a calibration gate before trusting any of our arms at
that scale.

**Anticipated cost, since this is where memory actually becomes a constraint:** 7B weights in bf16
are ~15 GB, and best-first search with `samples_per_step=32` issues 32 sequences per expansion. The
KV cache — not the weights — dominates, and it grows with prompt length, which retrieval directly
inflates by adding premises to context. vLLM's PagedAttention is the mitigation and an 80 GB A100
should hold it, but the pilot-before-committing rule applies: measure on 20 problems and extrapolate
before spending GPU-hours.

## Reproducing

### Requirements

Python 3.11, ~40 GB disk, and a GPU for index building (CPU works, ~28× slower).

```bash
git clone <this repo> && cd prooflens-prover
uv venv --python 3.11 && uv pip install -e '.[retrieval]'
```

### 1. Lean + Mathlib (once per machine)

```bash
scripts/setup_lean_project.sh ~/lean/mathlib_v4160 v4.16.0
```

Downloads pre-built `.olean` files — never compiles Mathlib. **v4.16.0 is not arbitrary**: all 141
FATE-M and 244 miniF2F statements elaborate on it, against 138 and 212 on v4.31.0, and a statement
that fails to elaborate is silently scored as unproved.

```bash
python scripts/lean_smoke.py --project-dir ~/lean/mathlib_v4160   # gate: must pass first
```

### 2. Premise corpus and indices

```bash
scripts/extract_premises.sh ~/lean/mathlib_v4160 data/premises/mathlib_v4160.jsonl
python scripts/build_bm25_index.py --corpus data/premises/mathlib_v4160.jsonl \
    --out data/index/bm25_mathlib_v4160

python scripts/build_dense_index.py --kind li --checkpoint <li-checkpoint> \
    --corpus data/premises/mathlib_v4160.jsonl --out data/index/li_ft_novel_bm25 \
    --assert-corpus-id 276070:31db61c63a9b7ee1
python scripts/build_dense_index.py --kind sv --checkpoint <sv-checkpoint> \
    --corpus data/premises/mathlib_v4160.jsonl --out data/index/sv_ft_novel_lr3e6 \
    --assert-corpus-id 276070:31db61c63a9b7ee1
```

`--assert-corpus-id` is what makes the arms comparable: it fails the build unless every retriever
ranks the identical candidate set. Without it a difference between arms could be a difference
between corpora.

The corpus is extracted from Lean's **elaborated environment**, not by parsing sources — macro-
generated declarations (`to_additive` alone accounts for thousands of citable lemmas) are invisible
to a regex.

### 3. Run an arm

```bash
python scripts/prove_benchmark.py --benchmark fate_m --arm li \
    --index data/index/li_ft_novel_bm25 --data-root <REAL-Prover>/data \
    --lean-project ~/lean/mathlib_v4160 --samples-per-step 32 --min-closers 19
```

Arms: `none` · `bm25` · `sv` · `li`. Everything except `--arm` is held fixed and recorded in the run
manifest, so two runs are comparable exactly when their manifests differ only in the arm.

`--min-closers 19` reserves candidate slots for the shared repertoire before premise tactics
compete. Without it, premise tactics displace closers and a *better* retriever can score *worse* —
observed as `none` 10/30 against `bm25` 6/30, where bm25's solved set was a strict subset of none's.

On SLURM:

```bash
BENCHMARK=fate_m ARM=li sbatch -p <gpu-partition> -A <account> -G 1 slurm/prove_benchmark.sbatch
```

### 4. Verify, then analyse

```bash
python scripts/verify_proofs.py --run results/logs/<run_id> \
    --data-root <REAL-Prover>/data --lean-project ~/lean/mathlib_v4160

python scripts/build_table1.py                       # discovers runs, emits the table above
python scripts/compare_arms.py --baseline results/logs/<sv_run> \
                               --treatment results/logs/<li_run>
```

`build_table1.py` discovers runs rather than taking ids, so a control cannot be paired against the
wrong benchmark's treatment.

To reproduce the published table from the exported records in this repo:

```bash
python scripts/build_table1.py --results-root results/exported/logs
```

### Tests

```bash
pytest tests/ -q -m "not lean"                                    # 288, hermetic
PROOFLENS_LEAN_PROJECT=~/lean/mathlib_v4160 pytest tests/ -q -m lean
```

---

## Design notes

**Search.** Best-first over the tactic tree, scoring `Σ log p(aₜ|sₜ) / depth^0.5` (REAL-Prover's
function, so budgets are comparable). Defaults match their shipped config: `max_expansions=64`,
`samples_per_step=16`, `max_depth=32`, 600 s per problem. States are deduplicated by pretty-printed
goal. The harness knows nothing about retrieval — an arm is a `TacticPolicy`.

**Lean.** [LeanInteract](https://github.com/augustepoiroux/LeanInteract) wrapping
`leanprover-community/repl`. The header environment is session-cached, so it survives the REPL
restarts `AutoLeanServer` performs under memory pressure; without that, one restart silently voided
32/141 problems in a run that otherwise completed and finalised normally.

**Late interaction.** Token embeddings in CSR layout; MaxSim via gather + `np.maximum.reduceat`.
Two-stage: mean-pooled candidate generation then exact MaxSim rerank, because exact full-corpus
MaxSim is ~2.3 GB of intermediates per query. `query_length=384` (26.4% of proof states exceed 256).

**Reproducibility.** Every run writes a manifest — config, seed, git SHA (with a `-dirty` marker),
package versions, Lean version, hardware, SLURM job id — before doing any work, so a job killed by
the scheduler still leaves a record of what it attempted.

## Layout

```
src/prooflens_prover/
  lean/         backend protocol, LeanInteract backend, proof verdicts
  retrieval/    base, bm25, dense (SV + LI), lean tokenizer
  prover/       best-first search, model-free tactic repertoire
  data/         benchmark and premise-corpus loaders
  eval/         paired comparison, significance
  utils/        seeding, logging, run manifests, io
scripts/        extraction, index building, benchmarks, verification, analysis
slurm/          cluster jobs
tests/          288 hermetic + 9 live-Lean
results/        exported run records and tables
```

## Acknowledgements

Benchmarks and reference numbers from [REAL-Prover](https://github.com/frenzymath/REAL-Prover)
(arXiv:2505.20613) and [FATE](https://github.com/frenzymath/FATE). Lean interaction via
[LeanInteract](https://github.com/augustepoiroux/LeanInteract). Retriever checkpoints from the
predecessor ProofLens premise-selection study.
