# ProofLens-Prover — retrieval architecture in a live Lean 4 theorem prover

A controlled study of **premise retrieval inside a working Lean 4 proof search**: does the retrieval
architecture change how many theorems a prover can close, and if so, where and why?

Every arm shares one search harness, one premise corpus, one Lean toolchain and one search budget.
Only the retriever varies. Every claimed proof is re-elaborated from the benchmark statement in a
fresh Lean environment before it is counted.

---

## Results

Retrieval arms over a **model-free** tactic policy — a fixed repertoire of 19 Lean tactics plus
templates that consume retrieved premises. There is no language model anywhere in this system.

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

On FATE-M, SV solved **35/141** against LI's **22/141**: −13 problems, p = 0.0023. This is the
opposite of the study's hypothesis and it is reported as measured. There is also real displacement —
15 problems only SV solved, 2 only LI solved — so the arms are not nested.

**However, the arms are not yet computationally matched, and the asymmetry favours SV:**

* **SV ranks all 276,070 premises exactly** — one matrix-vector product.
* **LI is two-stage** — mean-pooled vectors select the top 1,000 candidates (**0.36%** of the
  corpus), and MaxSim only ever sees those. A premise dropped by the first stage cannot be
  recovered by late interaction.

The `recall@10 = 0.992` printed at index-build time does **not** address this: it was measured with
*premise embeddings* as probes, so it describes a premise retrieving its neighbours, not a proof
state retrieving a premise. It is measured on the wrong distribution.

So the honest reading today is **"exact single-vector beats approximate multi-vector at a
first-stage budget of 0.36%"** — a claim about candidate generation, not about late interaction.
`scripts/measure_li_recall.py` settles which it is; see [Next steps](#next-steps).

### Cost

| | |
|---|--:|
| Premise corpus | 276,070 premises (Mathlib v4.16.0) |
| BM25 index | 169 MB, 9.8 ms/query |
| SV index | 943 MB, 768-dim |
| LI index | ~5.5 GB fp16, 21.7M token vectors (78.8/premise) |
| LI query | 77.9 ms |
| `import Mathlib`, NFS | 439–691 s |
| `import Mathlib`, node-local | **158 s** |

### Verification

`scripts/verify_proofs.py` re-elaborates every claimed proof from the benchmark statement in a fresh
Lean environment, sharing nothing with the search but the cheat-token regex. `sorry`, `admit` and
`native_decide` are rejected before execution and again at verification.

| run | verified |
|---|---|
| FATE-M / LI | 22/22 ✅ |
| FATE-M / none | ✅ |
| ProofNet / LI | ✅ |
| miniF2F / LI | 78/78 ✅ |
| **FATE-M / SV** | **not yet run** |

The SV row of the results table is therefore **provisional**. It is the number that reverses the
hypothesis and it has not been independently re-checked.

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

## Next steps

1. **Verify the SV run.** Nothing above is final until it passes.
2. **Measure LI's true first-stage recall** (`scripts/measure_li_recall.py`) at n_candidates =
   1,000 / 5,000 / 20,000 against exact full-corpus MaxSim, using real queries. If recall at 1,000
   is materially below 0.99, re-run the LI arm at the budget where it saturates. Until then the
   LI-vs-SV comparison is between an exact retriever and an approximate one.
3. **Re-run ProofNet and miniF2F LI** at `query_length=384`; those rows were produced at 256.
4. **Add SV on ProofNet/miniF2F** only if FATE-M's comparison survives step 2, since neither
   benchmark has shown resolution.
5. **A real generator.** 51 of 71 exhausted FATE-M searches hit `max_expansions`, not the clock —
   the search runs out of ideas, not time. No search budget substitutes for a language model.

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
