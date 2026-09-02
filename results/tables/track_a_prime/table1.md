# Table 1 — retrieval architecture, model-free policy (Track A')

| benchmark | none (floor) | ProofLens-SV | ProofLens-LI | Δ SV vs none | Δ LI vs none | Δ LI vs SV | SV ∪ LI (oracle) |
|---|--:|--:|--:|--:|--:|--:|--:|
| fate_m | 12/141 (8.5%) | 35/141 (24.8%) | 31/141 (22.0%) @50k | **+23** (p=0.0000) | **+19** (p=0.0000) | -4 (p=0.3438) | 38/141 (27.0%) [+3] |
| proofnet_test | 9/186 (4.8%) | 20/186 (10.8%) | 20/186 (10.8%) @50k | **+11** (p=0.0010) | **+11** (p=0.0010) | +0 (p=1.0000) | 24/186 (12.9%) [+4] |
| minif2f_test | 78/244 (32.0%) | 77/244 (31.6%) | 79/244 (32.4%) @50k | -1 (p=1.0000) | +1 (p=1.0000) | +2 (p=0.5000) | 79/244 (32.4%) [+0] |

**Δ vs none** is problems gained over the no-retrieval control; **Δ LI vs SV** is the
architecture comparison — same premise corpus, same search budget, same checkpoint
training protocol, only multi-vector against single-vector. Bold marks a result where the
bootstrap CI and the sign-flip permutation test agree.

**SV ∪ LI** is the oracle union: problems solved by *either* retriever, with the gain
over the better single arm in brackets. No single retriever can reach it, and it is not a
fusion result — it is the ceiling a fusion arm could reach. It sits above both arms only
because they disagree about *which* problems they solve, not about how many.

`@1k` / `@50k` on the LI cells is that run's **first-stage candidate budget**: LI
generates candidates with a mean-pooled vector, then reranks them with exact MaxSim, so
this is how many premises the exact stage ever sees. Measured recall@10 of that first
stage against exact full-corpus MaxSim is **0.443 at 1k** and **0.979 at 50k**. SV needs
no such annotation: it ranks all 276,070 premises exactly. An LI number at 1k is a lower
bound on the architecture, not a measurement of it.

Per-comparison detail — the exact problems each arm won, the displacement check, and premise-needed rates — is in `table1.json`.

## Published reference numbers

These come from systems built on 7B-class fine-tuned language models. The rows
above use a **model-free** tactic policy with no language model at all, so the
rates are not comparable. What Table 1 measures is the *effect of retrieval* with
the generator held fixed; the comparable-to-published experiment is Tier 1
(frozen REAL-Prover-v1).

* **fate_m** — REAL-Prover-v1 (7B, LeanSearch-PS): 56.7; REAL-Prover-v1 (no retrieval): 44.7
* **proofnet_test** — REAL-Prover-v1 (7B, LeanSearch-PS): 23.7; REAL-Prover-v1 (no retrieval): 22.6; ReProver (<1B): 13.8
* **minif2f_test** — REAL-Prover-v1 (7B): 54.1
