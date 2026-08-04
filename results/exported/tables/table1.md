# Table 1 — retrieval architecture, model-free policy (Track A')

| benchmark | none (floor) | ProofLens-SV | ProofLens-LI | Δ vs none | Δ vs SV |
|---|--:|--:|--:|--:|--:|
| fate_m | 12/141 (8.5%) | 35/141 (24.8%) | 22/141 (15.6%) | **+10** (p=0.0020) | **-13** (p=0.0023) |
| proofnet_test | 9/186 (4.8%) | — | 12/186 (6.5%) | +3 (p=0.2500) | — |
| minif2f_test | 78/244 (32.0%) | — | 78/244 (32.0%) | +0 (p=1.0000) | — |

**Δ vs none** is problems gained over the no-retrieval control; **Δ vs SV** is the
architecture comparison — same premise corpus, same search budget, same checkpoint
training protocol, only multi-vector against single-vector. Bold marks a result where the
bootstrap CI and the sign-flip permutation test agree.

Per-comparison detail — the exact problems each arm won, the displacement check, and
premise-needed rates — is in `table1.json`.

## Published reference numbers (context, not comparisons)

These come from systems built on 7B-class fine-tuned language models. The rows above
use a **model-free** tactic policy with no language model at all, so the rates are not
comparable. What Table 1 measures is the *effect of retrieval* with the generator held
fixed; the comparable-to-published experiment is Tier 1 (frozen REAL-Prover-v1).

* **fate_m** — REAL-Prover-v1 (7B, LeanSearch-PS): 56.7; REAL-Prover-v1 (no retrieval): 44.7
* **proofnet_test** — REAL-Prover-v1 (7B, LeanSearch-PS): 23.7; REAL-Prover-v1 (no retrieval): 22.6; ReProver (<1B): 13.8
* **minif2f_test** — REAL-Prover-v1 (7B): 54.1
