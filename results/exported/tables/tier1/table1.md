# Table 1 — retrieval architecture, frozen REAL-Prover-v1 (7B) (Tier 1)

| benchmark | none (floor) | ProofLens-SV | ProofLens-LI | Δ SV vs none | Δ LI vs none | Δ LI vs SV | SV ∪ LI (oracle) |
|---|--:|--:|--:|--:|--:|--:|--:|
| fate_m | 39/141 (27.7%) | 46/141 (32.6%) | 46/141 (32.6%) @50k | +7 (p=0.0923) | +7 (p=0.1671) | +0 (p=1.0000) | 57/141 (40.4%) [+11] |
| proofnet_test | 19/186 (10.2%) | 26/186 (14.0%) | 26/186 (14.0%) @50k | **+7** (p=0.0391) | **+7** (p=0.0391) | +0 (p=1.0000) | 32/186 (17.2%) [+6] |

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

Per-comparison detail — the exact problems each arm won, the displacement check — is in `table1.json`.

## Published reference numbers

The rows above hold **REAL-Prover-v1 (7B) frozen** and vary only the retriever, so
they are comparable in kind to these published numbers — but **not in budget**, and
the gap is enormous. REAL-Prover's figures are Pass@64x64: 64 passes of 64 nodes x
64 samples, about 4.2M generations per problem. Table 1 is a *single* pass at 64
nodes x 16 samples — 1,024 generations, roughly **1/4,000** of their budget. Read
any shortfall against 56.7 / 23.7 as a budget difference first and a system
difference second.

ReProver's ProofNet **13.8%** is the closest thing here to a like-for-like row: a
single pass from a sub-1B model with single-vector retrieval.

Premise attribution is **not reported** for these rows. 'Names a premise' is
decidable from proof text only for the model-free repertoire, whose tactics are
either a fixed closer or a premise template; every tactic a language model writes
is 'not a closer', so the same test would mark all of them and report a rate of
100% regardless of what retrieval contributed. See
`eval/compare.PREMISE_ATTRIBUTABLE_POLICIES`.

* **fate_m** — REAL-Prover-v1 (7B, LeanSearch-PS): 56.7; REAL-Prover-v1 (no retrieval): 44.7
* **proofnet_test** — REAL-Prover-v1 (7B, LeanSearch-PS): 23.7; REAL-Prover-v1 (no retrieval): 22.6; ReProver (<1B): 13.8
