# ProofLens-Prover — retrieval architecture in a live Lean 4 theorem prover

Does the **retrieval architecture** change how many theorems a prover can close?

Every arm shares one search harness, one premise corpus, one Lean toolchain and one search budget.
Only the retriever varies. Every claimed proof is re-elaborated from the benchmark statement in a
fresh Lean environment before it is counted.

## Headline

**An ensemble of both retrievers reaches the published state of the art at 1/128 of its generation
budget. The retrieval architecture still does not decide anything.**

| benchmark | this work (pass@8 ensemble) | REAL-Prover-v1 (Pass@64×64) | ReProver | budget |
|---|--:|--:|--:|--:|
| **ProofNet-test** | **44 / 186 = 23.7%** | 23.7% | 13.8% | **1/128 of theirs** |
| **FATE-M** | **72 / 141 = 51.1%** | 56.7% | — | **1/128 of theirs** |

32,768 generations per problem against their 4,194,304. On ProofNet that **matches** their published
figure to the precision it is reported at — 23.7% of 186 is 44.08 problems and the ensemble solves 44.
It does not beat it; a strict win needs 45. On FATE-M it falls 8 problems short.

The ensemble is two retrievers × eight seeds, accepting any proof any run finds — a deployable system,
not an oracle, and the same kind of object as a published Pass@64. It beats either single arm at
**every** k, including k = 1, and neither arm's curve has flattened by k = 8.

**And the architecture null gets stronger, not weaker, under 16× the budget:**

| | late interaction | single-vector | | |
|---|--:|--:|--:|---|
| ProofNet-test | **39** | **39** | 5 vs 5 exclusive | **p = 1.0000** |
| FATE-M | 68 | 66 | 6 vs 4 | p = 0.7539 |
| **pooled** | **107** | **105** | 11 vs 9 | **p = 0.8238**, CI [−2.14, +3.36] pts |

The original design predicted this was where the null *should* break: late interaction's advantage
lives in the mean candidate, and a wider search samples deeper into the ranking. Tested at 16× the
budget and eight independent draws, it holds — and ProofNet returns an exact tie for the second time.

---

**The single-draw results below stand as measured.** Retrieval works; the architecture does not decide
it. Two generators, chosen so their weaknesses do not overlap, agree — and they agree for different
reasons.

| | retrieval vs none | late interaction vs single-vector |
|---|---|---|
| **Tier 1** — frozen REAL-Prover-v1 (7B) | **+11 of 327** pooled — significant for SV (p = 0.038), **not** for LI (p = 0.089) | **+0 of 327** — 14 gained, 14 lost, **p = 1.0000** |
| **Track A′** — model-free policy | **+23 / +19** FATE-M (p < 0.0002), **+11** ProofNet (p = 0.0011) | −4 (p = 0.34), 0 (p = 1.00), +2 (p = 0.50) |

Late interaction costs **5.8× the index memory and 24× the query latency** to draw.

Two nulls, each with a mechanism — and the mechanisms are the contribution:

1. **Track A′ — the approximation, not the architecture.** Late interaction cannot rank a large
   corpus exactly, so it pre-selects candidates with a mean-pooled vector. At the conventional 1,000
   that first stage has **recall@10 = 0.443** against exact MaxSim, not the 0.992 an index build
   reports on the wrong query distribution. Widening it to 50,000 is worth **+9 on FATE-M and +8 on
   ProofNet** with nothing else changed — two thirds of what looked like an architecture difference.
2. **Tier 1 — the search reads the part retrieval does not improve.** At the root state, where every
   arm sees the identical theorem, retrieval measurably improves the generator — but late interaction
   improves the **mean candidate 4.9× and 13.1× more than the best**, and best-first search consumes
   only the best. Between the two retrieval arms the top candidate is *negative* on both benchmarks.
3. **The two arms fail in opposite ways at 16 samples — and identically at 32.** When LI loses it
   **always exhausts the full expansion budget**; when SV loses it dies far earlier, often silent. LI
   wins more *genuinely new* proofs (8 to 3 on FATE-M) but **breaks more that already worked** (7 of
   SV's 10 exclusive wins are problems the no-retrieval control also solved). Net: zero.
   **Corrected by the sweep (T10):** that asymmetry is a property of the 16-sample budget. Doubling
   samples cuts `no_candidates` on ProofNet from 42% of attempts to 19.5% and both arms then die at
   the expansion cap. A failure-mode claim about a proof search has to state its sample budget.
4. **The benchmarks barely exercise the regime LI wins in.** Its measured strength is
   out-of-distribution robustness, but the premises these proofs cite are **2.7× concentrated in the
   21.8% of Mathlib the retriever trained on**. The tie is what the predecessor's robustness result
   *predicts* once the deployment distribution is measured — not evidence against it.

Each retriever solves exactly 74 of 327 at one draw, but **14 differ in each direction, so
either-retriever reaches 88**. Equal counts, different theorems — and that gap is what the ensemble
above cashes in: run as a real system over eight seeds it reaches **116 of 327**, +28 on the
single-draw union and far more than retrieval's own +11 effect.

---

## The setup, in one page

**One harness, two generators**, behind the same `TacticPolicy` protocol — so search, benchmarks,
manifests, verification and significance are shared, not reimplemented.

| | Tier 1 | Track A′ |
|---|---|---|
| generator | **REAL-Prover-v1** — 7B, `Qwen2.5-Math-7B` base, stepwise, retrieval-native, held **frozen** | **no language model at all** — 19 fixed Lean tactics + 5 premise templates |
| retrieval is | **substitutive** — premises rewrite the prompt, so retrieval can *lose* a problem | **additive** — premises are extra candidates, so retrieval can only add |
| what it buys | comparable in kind to published systems | no prompt sensitivity, no sampling variance, no training confound |
| what it costs | trained on its own retriever's output, so ours are out-of-distribution for it | absolute rates are a floor, not a competitive result |

### The four arms

| arm | what it is | why it exists |
|---|---|---|
| **`none`** | **the floor.** No retriever is loaded and no query is issued (`n_queries: 0`). The generator sees only the proof state. | Every retrieval claim is measured *against* this. Without it, "84% of proofs cite a retrieved premise" is unfalsifiable — see [Contamination and attribution](#contamination-and-attribution). |
| **`bm25`** | lexical baseline, all 276,070 premises ranked exactly | cheapest possible retriever; Track A′ only |
| **`sv`** | **ProofLens-SV** — single-vector, one 768-dim embedding per premise, ranks all 276,070 **exactly** | the *matched control*: same encoder class, same training data, same corpus as LI. The only difference is the architecture. |
| **`li@50k`** | **ProofLens-LI** — late interaction (ColBERT-style), **one vector per token**, 21.7M vectors, two-stage: mean-pooled top-50,000 then exact MaxSim | the thing being tested |

`sv` is what makes this a controlled experiment rather than a demo. A retriever compared against
BM25 or against nothing tells you retrieval helps; a retriever compared against a *matched*
single-vector control tells you whether **late interaction** helps.

### What "the RAG" actually is here

Not document RAG. The retrieved units are **Lean premises** — every theorem, lemma, definition and
instance in Mathlib v4.16.0, extracted from Lean's *elaborated environment* rather than by parsing
source (macro-generated declarations like `to_additive` are invisible to a regex and account for
thousands of citable lemmas).

| | |
|---|--:|
| premise corpus | **276,070** premises, Mathlib v4.16.0 |
| corpus fingerprint | `276070:31db61c63a9b7ee1` — asserted at every index build, so all arms rank the identical set |
| query | the theorem statement at the root, the pretty-printed goal at every state thereafter |
| retrieved per query | top **10** premises, of which **6** fit the prompt budget |
| when | **at every proof state**, not once per problem — 3,874–5,631 queries per benchmark run |

Retrieval is re-run at every state deliberately: its leverage scales with how often it is queried,
and whole-proof generation invokes it once. For the LLM the premises are prepended to the proof state
in REAL-Prover's own prompt template; for the model-free policy they are substituted into five tactic
templates (`exact {p}`, `apply {p}`, `rw [{p}]`, `rw [← {p}]`, `simp [{p}]`).

### The search

Best-first, scoring `cumulative_logprob / max(n_tokens, 1)`. Identical for every arm:
`max_expansions=64`, `samples_per_step=16`, `max_depth=32`, `wall_clock=600 s`, `temperature=1.5`,
one pass, `seed=0`.

**Read the budget before the numbers.** REAL-Prover's published 56.7 / 23.7 are Pass@64×64 —
≈4.2M generations per problem. Every row here is a single pass at 1,024 generations, roughly
**1/4,000 of their budget**. Absolute rates are not comparable to theirs and are never presented as
if they were. What *is* controlled exactly is the retriever.

---

## Results — Tier 1: frozen REAL-Prover-v1 (7B)

Reported on the **node-local staged** run set, independently verified, and confirmed to span no
prover-code change — see [The staging confound](#the-staging-confound).

| benchmark | none | ProofLens-SV | ProofLens-LI @50k | Δ retrieval | Δ LI vs SV | SV ∪ LI (oracle) |
|---|--:|--:|--:|--:|--:|--:|
| **FATE-M** (141) | 39 (27.7%) | **46 (32.6%)** | **46 (32.6%)** | +7 | **+0** (10 vs 10) | **56 (39.7%)** [+10] |
| **ProofNet-test** (186) | 24 (12.9%) | **28 (15.1%)** | **28 (15.1%)** | +4 | **+0** (4 vs 4) | **32 (17.2%)** [+4] |
| **pooled** (327) | 63 (19.3%) | **74 (22.6%)** | **74 (22.6%)** | **+11** | **+0** (14 vs 14) | **88 (26.9%)** [+14] |

`scripts/build_table1.py --policy vllm`. Significance requires the paired bootstrap CI (10k) **and**
the sign-flip permutation test (10k) to agree; exact McNemar is reported since the outcome is binary
and paired.

### T1. The calibration gate, and what it actually showed

The plan's original gate (reproduce 56.7 within 3%) is unaffordable by 4,000× and would be failed by
a perfectly correct harness. It was replaced, **before running anything**, by three checks a correct
harness must pass at an affordable budget:

| gate | requirement | measured | |
|---|---|---|---|
| floor | the 7B must clear the model-free policy's FATE-M 22.0% | **32.6%** (46 vs 31) | ✅ |
| direction | retrieval must help, matching the sign of their 56.7 vs 44.7 ablation | **+11 pooled**, SV p = 0.038 | ✅ (weakened — LI's own contrast is not significant) |
| mechanics | non-degenerate generation | **12.1–12.8** distinct candidates of 16; empty rate **0.0001**; cheat rate **0.0**; logprob **−0.81 to −1.00** | ✅ |

One unplanned calibration fell out: **ReProver** — sub-1B, single-vector, stepwise, one pass —
reports **13.8% on ProofNet**; this harness reports **15.1%**. Evidence the rig is in the right
regime, not a comparison of systems.

### T2. Retrieval helps — but only pooled, and only for one arm

| contrast | FATE-M | ProofNet-test |
|---|---|---|
| SV vs none | +7, 10 gained / 3 lost, p = 0.0923 | +4, 8 / 4, p = 0.3877 |
| LI vs none | +7, 15 / 8, p = 0.2100 | +4, 8 / 4, p = 0.3877 |
| LI vs SV | +0, 10 / 10, p = 1.0000 | +0, 4 / 4, p = 1.0000 |

**No benchmark reaches significance on its own, for any contrast.** That is a statement about power,
and it can be made exactly: on 19 discordant pairs, exact McNemar cannot reach p < 0.05 for any split
closer than 15–4. Pooled across both benchmarks (327 problems, ids namespaced so no pair can span
them, one contrast at a time):

| contrast | Δ | gained / lost | 95% CI | p (perm) | p (McNemar) | verdict |
|---|--:|---|---|--:|--:|---|
| **SV vs none** | +11 | 18 / 7 | [+0.0031, +0.0642] | **0.0381** | 0.04329 | **SIGNIFICANT** |
| **LI vs none** | +11 | **23 / 12** | [+0.0000, +0.0703] | 0.0890 | 0.08953 | not significant |
| LI vs SV | **+0** | 14 / 14 | [−0.0306, +0.0306] | 1.0000 | 1.0000 | not significant |

Per-benchmark effects agree in sign in every contrast, the condition under which a pooled average
describes both benchmarks rather than neither. Sensitivity excluding harness errors changes no
verdict (p = 0.0429 / 0.0882 / 1.0000 on n = 322 / 323 / 322).

**Both retrieval arms gain the identical net +11 but get there differently.** SV gains 18 and loses
7; LI gains 23 and loses 12. **LI displaces nearly twice as many problems**, and that split — not the
net — is what costs it significance. Retrieval's prompt rewrite is costlier for the finer-grained
retriever. For the model-free policy displacement was exactly zero on both premise-heavy benchmarks,
because premises are *extra candidates* there and a *prompt rewrite* here. A study run on only one of
the two would have generalised the wrong way.

### T3. Why the architectures tie: the search reads the part retrieval does not improve

At **depth 0** every arm is prompted with the identical statement and differs only in the attached
premises — the one properly paired measurement available, since the arms diverge after their first
tactic and a state reached by one need not exist in the other's tree.

| root state | problems | candidates | mean logprob | **best** logprob |
|---|--:|--:|--:|--:|
| **FATE-M** | | | | |
| none | 141 | 1568 | −0.9332 | −0.1723 |
| SV | 141 | 1421 | −0.8566 | **−0.1426** |
| LI @50k | 141 | 1453 | **−0.8163** | −0.1465 |
| **ProofNet** | | | | |
| none | 182 | 2111 | −1.0606 | −0.1681 |
| SV | 182 | 2077 | −1.0318 | **−0.1498** |
| LI @50k | 182 | 2063 | **−0.9922** | −0.1612 |

| contrast | statistic | FATE-M | | ProofNet | |
|---|---|--:|---|--:|---|
| SV vs none | mean | **+0.1023** | p = 0.0034 ✅ | +0.0289 | p = 0.4315 |
| SV vs none | best | **+0.0297** | p = 0.0002 ✅ | **+0.0182** | p = 0.0226 ✅ |
| LI vs none | mean | **+0.1265** | p = 0.0002 ✅ | **+0.0905** | p = 0.0072 ✅ |
| LI vs none | best | **+0.0258** | p = 0.0038 ✅ | +0.0069 | p = 0.4618 |
| **LI vs SV** | mean | +0.0242 | p = 0.4689 | +0.0616 | p = 0.0939 |
| **LI vs SV** | **best** | **−0.0039** | p = 0.6091 | **−0.0113** | p = 0.2479 |

Three things follow:

1. **Retrieval genuinely improves the generator** — the mechanism behind the +11, measured where the
   comparison is clean. LI improves the mean candidate significantly on **both** benchmarks.
2. **It improves the mean candidate far more than the best.** For LI the mean gain is **4.9×** the
   best gain on FATE-M and **13.1×** on ProofNet. Best-first consumes candidates in rank order, so
   what decides a proof is the *top* — where retrieval's effect is smallest. The top candidate already
   sits at −0.14 to −0.17 per token (≈86% per-token probability); very little headroom is left.
3. **Between the architectures the top candidate is not better at all** — LI's edge is confined to
   the mean and is **negative on the best on both benchmarks**. There is no channel through which
   LI's advantage could become a proof: the search discards precisely the region where LI is ahead.

Two details run against the conclusion rather than for it. The control produced the *most* root
candidates on both benchmarks, and a maximum over more samples is stochastically larger, so the
control's best-candidate deficit is if anything understated; LI also drew more candidates than SV on
FATE-M and still lost on best.

**One honest wrinkle.** SV vs none shows the *opposite* asymmetry on ProofNet — best significant,
mean not. So mean-over-max is a property of the **LI arm specifically**, not a general law of
retrieval. That is consistent with the proof counts, where SV clears significance and LI does not.

An independent signal agrees: **undecodable** completions (containing U+FFFD, unparseable by Lean)
fall monotonically with retrieval quality. Collected for an unrelated reason, nothing to do with
log-probabilities.

So the 14/14 tie is not "the retrievers are equivalent". It is **the architecture difference is real,
is measurable in the generator, and sits below the resolution of the search** — the claim the
predecessor study hypothesised and could not test, having never run a prover.

### T4. Why the predecessor's robustness advantage does not transfer

The predecessor's headline was a robustness claim: on LeanDojo's `novel_premises` split, single-vector
dropped **−17.9%** from seen to novel premises while late interaction stayed flat and won 5/5 seeds —
and single-vector won on the easier in-distribution split. The architectures were never uniformly
ordered.

That predicts the tie is **two populations cancelling**: LI's wins should be the problems needing
premises the retriever never trained on. Tested against its own 62,500 training positives:

| exclusive wins, pooled | problems | premises cited/proof | fraction unseen |
|---|--:|--:|--:|
| only SV | 14 | 12.1 | 0.437 |
| only LI @50k | 14 | 8.2 | **0.451** |

Difference **+0.0139**, permutation **p = 0.8542**. **It is not two populations.** The reason is in
the levels:

| | none | SV | LI @50k | base rate |
|---|--:|--:|--:|--:|
| fraction of cited premises unseen | 0.412 | 0.405 | 0.408 | **0.782** |
| enrichment toward the training distribution | 2.70× | 2.73× | 2.72× | 1× |

The premises these benchmarks need are **2.7× concentrated** in the 21.8% of Mathlib the retriever
trained on. Not a coincidence and not a flaw: the retriever's training positives are the premises
human Mathlib proofs cite, and benchmark proofs cite the same commonly-used lemmas.

**The practical statement is conditional: choose multi-vector retrieval when the premise distribution
is genuinely novel relative to training — and standard Lean benchmarks are not that.**

### T5. The tie is not one run counted twice

Equal counts on two independent benchmarks is exactly the shape a duplicated or mislabelled run
takes. The decisive signal is proof text: at fixed code and environment the engine is deterministic,
so two runs of the *same* arm agree on every shared proof character for character.

| SV vs LI @50k | FATE-M | ProofNet-test | a duplicate would give |
|---|--:|--:|--:|
| both solved | 36 | 24 | — |
| **byte-identical proofs** | **16.7%** | **37.5%** | **~100%** |
| discordant problems | 20 | 8 | **0** |
| recorded retriever | `sv` / `li` | `sv` / `li` | identical |
| retrieval latency | 39.4 / 930.2 ms (23.6×) | 42.1 / 1014.9 ms (24.1×) | identical |

Four independent signals; a duplicate fails all four at once. **Verdict: distinct runs.**

**And the tie is unremarkable.** Given `k` disagreements between equivalent arms, an exact tie has
probability `C(k, k/2) / 2ᵏ` — **0.176** at FATE-M's 20 and **0.273** at ProofNet's 8. Those are the
*modal* outcomes: an exact tie is the single most likely result when two arms are equivalent, so
observing one is evidence **for** equivalence.

### T6. Equal counts, different theorems

| benchmark | SV | LI @50k | LI-only | SV-only | SV ∪ LI |
|---|--:|--:|--:|--:|--:|
| FATE-M | 46/141 | 46/141 | 10 | 10 | **56 (39.7%)** |
| ProofNet-test | 28/186 | 28/186 | 4 | 4 | **32 (17.2%)** |
| pooled | 74/327 | 74/327 | 14 | 14 | **88 (26.9%)** |

Perfect symmetry at every level — what a fair coin does, and the strongest form the null can take.
But **28 of the 88 problems in the union are solved by exactly one arm**, and the union sits **+14
above either arm while retrieval's whole measured effect is +11.**

The union is an **oracle** — it picks the winning retriever per problem, knowing the answer. Not a
result, a ceiling: the one a fusion arm chases, needing no new retriever and no training.

**And the disagreement is architectural, not noise.** Two runs of one arm on different nodes at fixed
code and environment are **byte-identical** — 46 vs 46 and 28 vs 28, discordance 0, 100% identical
proofs. The sampling noise floor is **zero**, so the 14/14 split is attributable to the retrievers.
(Across *different* environments it is not zero: FATE-M LI held 46 but moved four problems.)

### T7. How the two architectures fail — and it is not the way one would guess

`scripts/discordance_profile.py` asks, for each problem exactly one arm solved, **how the other arm
failed on it.** The distinction is load-bearing: `no_candidates` means the loser had nothing left to
propose, so retrieval's **recall** did the work; `max_expansions` means it had plenty to try and
spent its whole budget, so retrieval's **ranking** did.

| | FATE-M | ProofNet-test |
|---|---|---|
| **problems only SV solved** | 10 | 4 |
|  how LI failed | 6 `max_expansions`, 4 `no_candidates` | 3 `max_expansions`, 1 `no_candidates` |
|  expansions LI spent failing (median) | **64.0 — the cap** | **64.0 — the cap** |
|  what the win cost SV (its own median) | 12.5 expansions [3.0] | 6.5 [1.5] |
| **problems only LI solved** | 10 | 4 |
|  how SV failed | 6 `no_candidates`, 4 `max_expansions` | 2 each |
|  expansions SV spent failing (median) | **10.0** | **33.0** |
|  what the win cost LI (its own median) | 7.0 [3.0] | **1.0 [1.0]** |

1. **When LI loses, it always dies at the expansion cap** — median 64 of 64 on both benchmarks. **LI
   never runs out of things to try; it runs out of budget.** SV dies far earlier, often silent.
2. **Difficulty runs opposite to intuition.** **SV's exclusive wins are its own hardest proofs** —
   4× its median search — while **LI's ProofNet wins are its easiest**: 1 expansion, a 1-step proof.
   LI picks up problems SV never surfaced a usable premise for, not problems needing deeper search.
3. **LI finds more new proofs and breaks more old ones.** Of SV's 10 exclusive FATE-M wins, **7 were
   also solved by the control** — those are problems **LI lost that the baseline already had**. Of
   LI's 10, only 2. Counting genuinely new proofs, **LI wins 8 to SV's 3**.

> Late interaction is the more **productive** retriever and the more **destructive** one. Its finer
> ranking surfaces premises single-vector never offers — rescuing problems from silence and winning
> more genuinely new theorems — but it also fills the frontier with plausible-but-wrong premises, so
> when it is wrong the search wanders until the budget is gone. Net: zero.

That is exactly what T3 predicts from the other direction: LI's advantage is in the **mean**
candidate — a broad, uniformly-plausible frontier — which is precisely the shape that rescues a
silent search and starves a directed one. **And it says where the gain is: not better ranking, but
not losing what already worked.**

**It is not about the mathematics.** Classifying each exclusive win by the typeclasses its statement
mentions, both arms' win sets mirror the benchmark's own composition (FATE-M is 57% group theory,
25% ring/field; SV's 10 wins are 5/3, LI's are 6/2). No subject-matter signal.

**Sample sizes forbid inference.** These sets are 4–10 problems; no p-value is computed and the
script refuses to print one.

### T8. Where the searches die

| | proved | `no_candidates` | `max_expansions` | wall clock | error |
|---|--:|--:|--:|--:|--:|
| FATE-M / none | 39 (28%) | 46 (33%) | 56 | 0 | 0 |
| FATE-M / SV | 46 (33%) | 44 (31%) | 51 | 0 | 0 |
| FATE-M / LI | 46 (33%) | 40 (28%) | 55 | 0 | 0 |
| ProofNet / none | 24 (13%) | 78 (42%) | 79 | **1** | 4 |
| ProofNet / SV | 28 (15%) | 77 (41%) | 76 | 0 | 5 |
| ProofNet / LI | 28 (15%) | 79 (42%) | 75 | 0 | 4 |

**The wall clock binds exactly once in 981 problem-arms.** Whatever limits this prover, it is not
time per problem — measured rather than argued.

**28–42% of problems end in `no_candidates`** — the generator has nothing usable left to propose.
That is ProofNet's largest failure mode, larger than the expansion cap. **Retrieval shrinks it on
FATE-M** (46 → 44 → 40, LI lowest) but **not on ProofNet** (78 → 77 → 79), consistent with
retrieval's weaker measured effect there.

### T9. Cost

| | wall clock | queries | ms/query | time in retrieval |
|---|--:|--:|--:|--:|
| FATE-M / none | 0.83 h | 0 | — | — |
| FATE-M / SV | 0.98 h | 3,874 | **39.4** | 0.04 h (4%) |
| FATE-M / LI @50k | **2.08 h** | 4,222 | **930.2** | **1.09 h (52%)** |
| ProofNet / none | 1.78 h | 0 | — | — |
| ProofNet / SV | 1.94 h | 5,472 | **42.1** | 0.06 h (3%) |
| ProofNet / LI @50k | **3.45 h** | 5,631 | **1,014.9** | **1.59 h (46%)** |

**24× the query latency and 1.8–2.1× the wall clock**, for 14 problems gained and 14 lost. **LI
spends roughly half of its entire run inside the retriever** where SV spends 3–4%. Both returned a
mean of exactly 10.0 premises per query, so the prompt budget was identical and the difference is
ranking alone; no prompt was ever truncated (1,536 tokens seen against a 3,840 limit), which removes
truncation as a confound.

**On this corpus, at this scale, with either generator, multi-vector retrieval does not pay for
itself.**

---

## Results — pass@8: the two-retriever ensemble

Everything above is one draw per arm at REAL-Prover's shipped budget (64 nodes x 16 samples). That
design answers the architecture question and nothing else: one seed carries no estimate of its own
variance, and the union of 88 above is an **oracle ceiling**, since picking the winning retriever per
problem requires knowing the answer.

This section removes both limits. **Eight independent seeds per arm, 32 samples per expansion, 25% of
each expansion sampled without premises** — 32 runs, ~124 GPU-hours — reported as a *deployable
ensemble*: run both arms at every seed and accept any proof any of them finds. Nothing needs to know
in advance which arm wins, which is what makes it a system rather than an oracle.

### T10. The coverage curve

pass@k by the unbiased estimator `1 - C(K-c,k)/C(K,k)` over **seed** subsets (Chen et al. 2021), not
the union of the k runs that happened to be first — that would count the lucky ordering.

**ProofNet-test** (186)

| arm | @1 | @2 | @4 | @6 | @8 |
|---|--:|--:|--:|--:|--:|
| late interaction | 17.7 | 18.9 | 19.8 | 20.4 | 21.0 |
| single-vector | 17.7 | 19.5 | 20.5 | 20.8 | 21.0 |
| **ensemble** | **19.8** | 21.0 | 22.2 | 23.0 | **23.7** |

**FATE-M** (141)

| arm | @1 | @2 | @4 | @6 | @8 |
|---|--:|--:|--:|--:|--:|
| late interaction | 38.1 | 42.5 | 46.0 | 47.4 | 48.2 |
| single-vector | 37.1 | 40.8 | 44.3 | 45.9 | 46.8 |
| **ensemble** | **42.0** | 45.8 | 48.7 | 50.1 | **51.1** |

| | single draw | **pass@8 ensemble** | gain |
|---|--:|--:|--:|
| ProofNet-test | 32 / 186 (17.2%) | **44 / 186 (23.7%)** | **+12** |
| FATE-M | 56 / 141 (39.7%) | **72 / 141 (51.1%)** | **+16** |
| pooled | 88 / 327 (26.9%) | **116 / 327 (35.5%)** | **+28** |

The ensemble beats either arm at **every** k, and **neither single arm has flattened by k = 8** —
ProofNet LI still gains 0.3 points from k = 7 to k = 8. The ceiling here is a budget, not the method.
Extending to pass@16 needs 32 more independent jobs and no rerun.

### T11. The null survives 16x the budget

The design's own prediction was that this is where the null should break: late interaction's advantage
lives in the **mean** candidate (T3), and a wider search samples further down the ranking, so a larger
budget should convert more of it into proofs.

| | LI | SV | li only | sv only | shared | McNemar |
|---|--:|--:|--:|--:|--:|--:|
| ProofNet-test | **39** | **39** | 5 | 5 | 34 | **p = 1.0000** |
| FATE-M | 68 | 66 | 6 | 4 | 62 | p = 0.7539 |
| **pooled** | **107** | **105** | 11 | 9 | 96 | **p = 0.8238** |

Pooled effect **+2 of 327 (+0.61 pts)**, paired bootstrap 95% CI **[-2.14, +3.36]**, sign-flip
permutation **p = 0.8178** — the CI and the permutation test agree, which is this project's gate for
reporting either. ProofNet returns **exactly 39 against 39** for the second time.

The null now holds across **two Lean environments, two generators, two search budgets and eight seeds
per arm**. That is the form the result should be quoted in.

### T12. T7 was a sample budget, not a mechanism

T7 above is the most striking mechanism claim in the single-draw results: when LI loses it always
exhausts the expansion budget (median 64 of 64) while SV dies early and silent (10 and 33). **At 32
samples it disappears.**

| | how the loser failed | median expansions when exhausted |
|---|---|--:|
| ProofNet, 5 solved only by LI | SV: 25 exhausted, 14 no_candidates | **64 of 64** |
| ProofNet, 5 solved only by SV | LI: 31 exhausted, 9 no_candidates | **64 of 64** |
| FATE-M, 6 solved only by LI | SV: 39 exhausted, 9 no_candidates | **64 of 64** |
| FATE-M, 4 solved only by SV | LI: 25 exhausted, 7 no_candidates | **64 of 64** |

Why: **`no_candidates` on ProofNet fell from 42% of attempts to 19.5%.**

| | exhausted | no_candidates | proved | error |
|---|--:|--:|--:|--:|
| ProofNet LI (1,488 attempts) | 59.8% | **19.5%** | 17.7% | 3.0% |
| ProofNet SV (1,488 attempts) | 60.1% | **19.2%** | 17.7% | 3.0% |
| FATE-M LI (1,128 attempts) | 47.8% | 13.5% | 38.1% | 0.6% |
| FATE-M SV (1,128 attempts) | 50.3% | 12.6% | 37.1% | 0.1% |

SV was not dying early because single-vector retrieval leaves problems *unreachable*. It was dying
early because 16 samples per expansion were not enough to keep the frontier alive. So T7 describes the
16-sample regime and must not be generalised past it; what survives is weaker and more robust — given
enough samples to keep searching, the two arms fail identically, and the ensemble gain comes from
*which* problems each reaches.

The identical 3.0% error rate on both arms also closes a loose end from the single-draw set, where
ProofNet SV carried one more harness error than LI and left open whether the ProofNet tie was really
SV +1.

### T13. Verification, and one rejected proof

All 32 runs re-elaborated from their benchmark statements in a fresh, node-local Lean environment.
**1,376 claims, 1 rejected.**

| | runs | claims | rejected |
|---|--:|--:|--:|
| ProofNet late interaction | 8 | 264 | 0 |
| ProofNet single-vector | 8 | 264 | **1** |
| FATE-M late interaction | 8 | 430 | 0 |
| FATE-M single-vector | 8 | 418 | 0 |

The rejected claim is worth reporting because of what it was:

```
proof[0] = 'let'
proof[1] = 'exact Nat.zero_lt_succ 999'
-> invalid binder name 'Nat.zero_lt_succ', it must be atomic
```

A bare `let` as the first step. During search each tactic is applied to a proof state on its own, so
`let` passed as one step; verification joins the steps with newlines, where `let` swallows the next
line as its binder name and the block stops parsing. **The rejection is correct** — a proof that does
not elaborate is not a proof — but it is a *serialisation* defect, not a `sorry` and not an unsound
step.

That problem is **discounted**: it never enters the solved set, stays in the denominator, and its proof
is not retained. The run is kept. Discarding the whole run would look stricter and be worse — that
seed holds the joint-highest count of its arm, so dropping it removes a high seed from an eight-seed
estimate and biases the arm further than the single claim being corrected. Discounting can only lower
a rate. It cost nothing here: the same problem was solved by a late-interaction seed whose proof
verified.

---

## Results — Track A′: model-free policy

No generator, so any difference between arms is the retriever alone. **Unaffected by the staging
issue** — these runs never used the LLM path.

| benchmark | none (floor) | ProofLens-SV | ProofLens-LI @50k | Δ SV vs none | Δ LI vs none | Δ LI vs SV | SV ∪ LI |
|---|--:|--:|--:|--:|--:|--:|--:|
| **FATE-M** (141) | 12 (8.5%) | **35 (24.8%)** | 31 (22.0%) | **+23** (p<0.0001) | **+19** (p=0.0002) | −4 (p=0.34) | **38 (27.0%)** [+3] |
| **ProofNet-test** (186) | 9 (4.8%) | **20 (10.8%)** | **20 (10.8%)** | **+11** (p=0.0011) | **+11** (p=0.0011) | 0 (p=1.00) | **24 (12.9%)** [+4] |
| miniF2F-test (244) | 78 (32.0%) | 77 (31.6%) | 79 (32.4%) | −1 (p=1.00) | +1 (p=1.00) | +2 (p=0.50) | 79 (32.4%) [+0] |

### A1. Retrieval helps only where premises are needed

On both premise-heavy benchmarks the control's solved set is a **strict subset** of the retrieval
arms': zero displacement, retrieval never cost a problem. The effect tracks whether the benchmark
*requires* citing lemmas — **+19 to +23** on graduate abstract algebra, **+11** on undergraduate
mixed, **+1 of 244** on competition arithmetic, where problems fall to `linarith` / `omega` /
`norm_num` and retrieval has no leverage.

### A2. "Premise used" is not "premise needed"

| benchmark | used (of proofs found) | **needed** (of proofs the control could not find) |
|---|--:|--:|
| FATE-M / SV | 86% | **23/23 = 100%** |
| FATE-M / LI | 84% (26/31) | **19/19 = 100%** |
| ProofNet / SV | 80% | **11/11 = 100%** |
| ProofNet / LI | 85% (17/20) | **11/11 = 100%** |
| miniF2F | 23% (18/79) | 1/1 |

Only the second is causal. The intuitive metric overstated retrieval's contribution by 60% on
FATE-M. Distinguishing them requires a paired control; reading proof text cannot do it.

### A3. The approximation was the effect, not the architecture

Late interaction cannot rank a large corpus exactly, because its defining property is keeping **one
vector per token**:

| | vectors stored | index size | exact search |
|---|--:|--:|---|
| ProofLens-SV | 276,070 (one per premise) | 943 MB | one matrix-vector product — **feasible** |
| ProofLens-LI | **21,752,080** (78.8 per premise) | 5.5 GB fp16 | ≈**8.4 billion floats (33 GB) per query** — **infeasible** |

So LI is necessarily two-stage, and a premise the first stage drops cannot be recovered however good
the late interaction is. SV has no such stage. The `recall@10 = 0.992` printed at index-build time
did **not** license ignoring this — it was measured with *premise embeddings* as probes, a premise
retrieving its neighbours, not a proof state retrieving a premise. Wrong distribution.

Measured on 141 real FATE-M queries against exact full-corpus MaxSim:

| n_candidates | % of corpus | recall@10 | queries lossless |
|--:|--:|--:|--:|
| **1,000** (the original default) | 0.36% | **0.443** | 9/141 |
| 5,000 | 1.81% | 0.696 | 32/141 |
| 20,000 | 7.24% | 0.888 | 81/141 |
| **50,000** (used above) | 18.11% | **0.979** | 124/141 |

The index-build figure overstated recall by **2.2×**. Widening the first stage from 1,000 to 50,000
is worth **+9 on FATE-M** (22/141 → 31/141, CI [+0.028, +0.106], p = 0.0028, McNemar 0.0039, zero
displacement) **and +8 on ProofNet** (12 → 20) with one integer changed.

**Two thirds of what looked like a significant "single-vector wins" result was an artefact of a
default.** The remaining 4 problems are noise (p = 0.34). ProofNet replicates it independently on a
different distribution — its recall curve was never measured, so the effect size there was a
prediction, not a fit.

### A4. Equal counts, different theorems

| benchmark | SV | LI @50k | LI-only | SV-only | SV ∪ LI |
|---|--:|--:|--:|--:|--:|
| FATE-M | 35/141 | 31/141 | 3 | 7 | **38 (27.0%)** |
| ProofNet-test | 20/186 | 20/186 | 4 | 4 | **24 (12.9%)** |
| miniF2F-test | 77/244 | 79/244 | 2 | 0 | 79 (32.4%) |

### A5. Cost

| | size | ms/query | ranks |
|---|--:|--:|---|
| BM25 | 169 MB | **9.8** | all 276,070 exactly |
| ProofLens-SV (768-dim) | 943 MB | **41.8–44.7** | all 276,070 exactly |
| ProofLens-LI @1k | ~5.5 GB | 76.9–85.9 | 0.36% of corpus, recall 0.443 |
| ProofLens-LI @50k | ~5.5 GB | **1030.9–1128.0** | 18.1% of corpus, recall 0.979 |

Latency scales with the rerank, not the index: 50× the candidates costs 13× the wall clock.

For scale context: REAL-Prover's own retriever `LeanSearch-PS` is built on
`intfloat/e5-mistral-7b-instruct` — a **7B single-vector** embedding model. Both retrievers here are
ModernBERT-base class (~149M), roughly **47× smaller**.

---

## Contamination and attribution

The retrieval corpus is all of Mathlib, and some benchmark theorems are restatements of lemmas
Mathlib already contains — ProofNet is transcribed from textbook exercises Mathlib also formalises.
When that happens the retriever can hand over the theorem itself and the proof closes in one step
(`exact Sylow.normalizer_normalizer P`). That is a **valid Lean proof**, not a `sorry`-style cheat, so
nothing in the Lean backend flags it. The only defence is measuring it.

`scripts/contamination_audit.py` counts solved problems whose **entire proof is a single `exact` or
`apply` naming a corpus premise** — the narrowest possible definition, so it under-counts, which is
the right direction for a figure whose job is to bound a concern:

| | solved | 1-step corpus answer | of solved | won vs control | **of those, 1-step** |
|---|--:|--:|--:|--:|--:|
| FATE-M / none | 39 | 3 | 7.7% | — | — |
| FATE-M / SV | 46 | 10 | 21.7% | 10 | **5** |
| FATE-M / LI @50k | 46 | 7 | 15.2% | 15 | **2** |
| ProofNet / none | 24 | 2 | 8.3% | — | — |
| ProofNet / SV | 28 | 1 | 3.6% | 8 | **0** |
| ProofNet / LI @50k | 28 | 4 | 14.3% | 8 | **0** |

**7 of the 41 problems retrieval won — 17% — closed by a single corpus citation**, and every one of
them is on FATE-M. **On ProofNet, not a single problem retrieval won was a one-step corpus answer.**
The other 83% required the prover to do genuine work with the premise.

* **It cannot manufacture an architecture difference.** Both arms index the identical corpus
  (`--assert-corpus-id` enforces it at build time). Corpus overlap inflates absolute pass rates and
  part of retrieval-vs-none; it is neutral between SV and LI, which is the contrast this study is
  about.
* **This is the intended mechanism at its degenerate limit.** Premise retrieval exists to supply
  lemmas the prover cannot derive. It is only a problem if it is the *whole* effect, and it is a
  sixth of it.
* **The control bounds memorisation** — `none` closes 3 and 2 the same way with no retriever present.
* **Generator contamination is unmeasured and affects all arms equally.** REAL-Prover-v1's training
  data is not fully public. It would inflate every row including `none`, and cancels in every
  contrast reported.

**Attribution is withheld for the LLM.** The Track A′ "premise used" metric is decidable from proof
text only when every tactic is either a fixed closer or a premise template. Applied to *generated*
tactics it would report ~100% regardless of what retrieval contributed — a confident, meaningless
number (`eval/compare.PREMISE_ATTRIBUTABLE_POLICIES`).

---

## Verification

`scripts/verify_proofs.py` re-elaborates every claimed proof from the benchmark statement in a fresh
Lean environment, sharing nothing with the search but the cheat-token regex. `sorry`, `admit` and
`native_decide` are rejected before execution and again at verification.

**Every run reported here verifies clean.** Every reported proof re-elaborates from scratch; the
tables are not provisional.

### The staging confound

`import Mathlib` reads ~5.4 GB of `.olean` files: **158 s** node-local against **439–691 s** from
NFS. That was treated as an optimisation, and the fallback message claimed NFS was "slower warm-up,
still correct". **Both were wrong.** On ProofNet, same arm, same seed:

| | proved | errors | REPL restarts |
|---|--:|--:|--:|
| node-local | **28** | 4 | 0 |
| NFS | 26 | 6 | 2 |

**The penalty is uneven across arms, and it hit the control hardest.** ProofNet `none` moved 19 → 24
against +2 for each retrieval arm, so **the earlier NFS run set overstated retrieval's effect**: the
pooled figure fell from +14 to +11 and LI vs none crossed from p = 0.0124 to p = 0.0890. FATE-M is
insensitive (39 / 46 / 46 either way).

Both sbatch scripts now **abort** rather than fall back to NFS silently (`ALLOW_NFS_FALLBACK=1` to
override, and record that the run is not comparable). All Tier 1 numbers above are node-local.

The staged run set spans two commits (`d7fb3c34` and `ab99690c`); `git diff` over `src/`, `scripts/`
and `configs/` shows **1,122 insertions, zero deletions, all four files analysis-only** — nothing on
the proving path changed, so the arms are comparable.

### Run records

`results/exported/` holds **36 runs** — each manifest and per-problem outcome, with per-tactic traces
stripped. Both tables regenerate from these records alone:

```bash
python scripts/build_table1.py --results-root results/exported/logs --policy vllm
python scripts/build_table1.py --results-root results/exported/logs --policy repertoire
```

Superseded runs are kept rather than deleted, because several are the evidence for defects described
here — the `n_candidates=1000` FATE-M run is the baseline for
[A3](#a3-the-approximation-was-the-effect-not-the-architecture), the NFS runs are the evidence for
the staging confound, and the 0/5 Tier 1 smoke run under the wrong chat template against 2/5 under
ChatML is the whole evidence for the prompt-format finding. `build_table1.py` selects the most recent
*finalised* run per (benchmark, arm), so their presence affects nothing.

---

## Conclusions

1. **Runtime premise retrieval measurably improves a live Lean prover, but the effect is smaller and
   more fragile than a single run set suggested.** Frozen 7B: **+11 of 327** pooled, significant for
   single-vector (p = 0.038) and **not** for late interaction (p = 0.089) — the difference being
   displacement, not gain. Model-free: +19 to +23 and +11, with zero displacement.
2. **It holds only where proofs require citing lemmas the policy cannot otherwise reach.** On
   competition arithmetic the effect is one problem in 244. **Benchmark choice determines whether a
   retrieval claim is measurable at all.**
3. **For the retrieval *architecture*, the answer is a null**, under both generators and on every
   benchmark, at 5.8× the memory and 24× the latency.
4. **The null has two distinct mechanisms, and they are the contribution.** Model-free: the
   **two-stage approximation**, worth +9 and +8 once corrected. Frozen 7B: **resolution** — LI
   improves the mean root candidate 4.9× and 13.1× more than the best, and best-first consumes only
   the best.
5. **The two architectures fail in opposite ways at 16 samples — and identically at 32.** LI always
   exhausts the full expansion budget when it loses; SV dies early and silent. LI wins more genuinely
   new proofs (8 to 3 on FATE-M) but breaks more that already worked (7 of SV's 10 exclusive wins were
   control-solved). **Corrected by [T12](#t12-t7-was-a-sample-budget-not-a-mechanism):** the asymmetry
   is a property of the sample budget, not of the architectures. A failure-mode claim about a proof
   search must state the sample budget it was measured at.
6. **The tie is conditional, not general.** These benchmarks' proofs are 2.7× concentrated in the
   retriever's training distribution, so the regime LI wins in is barely exercised.
7. **A null on the counts is not a null on the theorems.** Each retriever solves 74 of 327 while 28
   problems are solved by exactly one; the union reaches 88. The fusion ceiling (+14) exceeds
   retrieval's entire measured effect (+11).
8. **Retrieval attribution needs a paired control** — the intuitive metric overstated the causal
   contribution by 60%, and under an LLM it becomes unmeasurable and is withheld.
9. **Any late-interaction result should state its first-stage recall on the real query
   distribution.** The plausible 0.992 available at index-build time overstates it by 2.2×.
10. **Whether retrieval can *cost* a proof is a property of the policy, not of retrieval** — zero
    displacement for the repertoire, 7–12 problems lost for the LLM.
11. **The execution environment is a first-class experimental variable.** A shared filesystem changed
    a published number by 5 problems on one arm and 2 on another. Treating infrastructure as an
    optimisation rather than a controlled condition produced the largest single correction here.
12. **The ensemble reaches the published state of the art at 1/128 of its budget.** ProofNet-test
    **44 of 186 (23.7%)** at pass@8, level with REAL-Prover-v1's Pass@64×64 on 32,768 generations per
    problem against 4,194,304; FATE-M **72 of 141 (51.1%)** against their 56.7%. The +14 union above
    was an oracle ceiling; run as a real two-arm ensemble over eight seeds it is worth **+28 of 327**,
    and it beats either arm at every k including k = 1.
13. **The architecture null is the project's strongest claim, not its weakest.** It survives a **16×
    budget increase** and eight independent draws — pooled 107 vs 105, p = 0.8238, CI [−2.14, +3.36];
    ProofNet **exactly 39 against 39, p = 1.0000**. T3 predicted a wider search would be where it
    broke. It does not.
14. **A sample budget can masquerade as a mechanism.** Doubling samples cut ProofNet's `no_candidates`
    from 42% of attempts to 19.5% and erased T7's asymmetry entirely. The most interesting mechanism
    in the single-draw results was an artefact of a budget, and only a second budget revealed it.

## Limitations

**Tier 1 (frozen 7B)**

* Budget is **~1/4,000** of the published configuration. Absolute rates are not comparable to
  56.7 / 23.7. The *contrast* is unaffected — every arm has the identical budget.
* **The model was trained on its own retriever's output**, so both ProofLens retrievers are
  out-of-distribution for it. This makes retrieval-vs-none conservative but **weakens the architecture
  null**: LI might be handicapped by distribution shift rather than by architecture. A
  retriever-agnostic generator is the clean test and has not been run.
* **The null is a null at one budget.** T3 shows LI's advantage lives in the mean rather than the
  max, so a *wider* search is exactly the regime where it might break. The design makes that
  prediction and does not test it.
* **No benchmark is individually significant**, for any contrast, and the calibration argument now
  rests on the pooled SV row alone.
* **Determinism is conditional** — byte-identical at fixed code and environment, not across
  environments: FATE-M LI held 46 but moved four problems.
* `proofnet_test sv` carries one REPL restart and one more error than LI, so even staged the arms did
  not run under quite identical conditions. If that error is transient, ProofNet is SV +1.
* The `none` arm's manifest records an `index` it never opened; `policy_config.retriever: none` and
  `n_queries: 0` are the fields that say what happened. Cosmetic, but it reads like a mislabelled run.
* **The unseen-premise metric has three known weaknesses**, reported rather than corrected: names are
  matched across two Mathlib versions and **2,331 of 62,500 (3.7%)** have no v4.16.0 equivalent; an
  abbreviated citation can denote several premises and the tie is broken toward *seen*, biasing
  **against** the finding sought; and whether tactic keywords count changes the level though not the
  verdict.
* The T4 and T7 analyses rest on 4–14 problems per cell. They can exclude a large effect, not a small
  one, and T7 computes no p-value by design.
* The depth-0 analysis paired 182 of 186 ProofNet problems — the 4 statements that do not elaborate
  under v4.16.0.

**Track A′ (model-free)**

* Absolute rates are **not** comparable to 7B-class systems, and the tactic priors are hand-set.
* **On miniF2F, SV solves 77 against the control's 78** — retrieval cost a problem. Not significant at
  n = 244, but the right sign for the mechanism.
* 4 ProofNet statements do not elaborate under v4.16.0; the effective denominator is 182. 186 is
  reported for comparability with published numbers.
* The oracle union is an upper bound obtained by knowing which arm wins. A real fusion arm would land
  below it.
* LI's recall curve was measured on FATE-M only.
* Retrieval queries the *statement* at the root and the pretty-printed goal thereafter; no proof
  context beyond the goal is used.

---

## Next steps

| phase | question | cost | status |
|---|---|---|---|
| 1 | Does LI win the problems needing premises it never trained on? | analysis only | ✅ **answered — no** ([T4](#t4-why-the-predecessors-robustness-advantage-does-not-transfer)) |
| 2 | Is the +0 stable, and is the disagreement above noise? | analysis only | ✅ **answered — noise floor is zero at fixed code and environment** ([T6](#t6-equal-counts-different-theorems)) |
| 2b | ProofNet under uniform node-local staging | 4.6 GPU-h | ✅ **done** — retrieval's effect fell from +14 to +11 |
| 3 | Does LI pull ahead at wider search? (T3 predicts the null breaks in the mean, not the max) | 124 GPU-h | ✅ **answered — no** ([T11](#t11-the-null-survives-16x-the-budget)); ProofNet returns an exact 39–39 tie |
| 3b | Does the ensemble reach the published numbers? | (same runs) | ✅ **ProofNet yes, at 1/128 the budget; FATE-M 8 problems short** ([T10](#t10-the-coverage-curve)) |
| 4 | Does the benchmark-dependence hold for an LLM? (miniF2F under Tier 1) | ~9 GPU-h | not started |
| **5** | **Does fusing the two rankings beat both, rather than unioning them?** | ~15 GPU-h | mechanism proven (3-problem smoke), not measured |
| 6 | **pass@16** — neither arm's curve has flattened at k = 8 | ~124 GPU-h | not started, needs no rerun |

**Phase 5 is still the strongest lead, and T12 sharpens what it has to beat.** The ensemble buys its
+28 by paying for both retrievers at every seed. A *fused* retriever would have to reach the same
problems on one retrieval budget. `FusionRetriever` (reciprocal-rank fusion, K = 60) is implemented and
smoke-tested; it has never been measured at scale.

**Phase 6 is the cheapest real gain.** ProofNet LI still gains 0.3 points from k = 7 to k = 8, so the
23.7% is a budget ceiling rather than a limit of the method — and 32 more independent jobs extend it
with no rerun of anything.

**Deferred rather than sequenced:** Track B (LoRA-trained generators) and PutnamBench. Both are in the
original plan and both are now poor value — Track B adds the training confound the frozen-model design
exists to avoid, and PutnamBench cannot discriminate between retrievers at this scale.

---

## Reproducing

Python 3.11, ~40 GB disk, a GPU for index building (CPU works, ~28× slower).

```bash
git clone <this repo> && cd prooflens-prover
uv venv --python 3.11 && uv pip install -e '.[retrieval]'
```

**1. Lean + Mathlib** (once per machine)

```bash
scripts/setup_lean_project.sh ~/lean/mathlib_v4160 v4.16.0
python scripts/lean_smoke.py --project-dir ~/lean/mathlib_v4160   # gate: must pass first
```

**v4.16.0 is not arbitrary**: all 141 FATE-M and 244 miniF2F statements elaborate on it, against 138
and 212 on v4.31.0 — and a statement that fails to elaborate is silently scored as unproved.

**2. Premise corpus and indices**

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
ranks the identical candidate set. Without it, a difference between arms could be a difference
between corpora.

**3. Run an arm**

```bash
# Track A' (model-free)
python scripts/prove_benchmark.py --benchmark fate_m --arm li \
    --index data/index/li_ft_novel_bm25 --data-root <REAL-Prover>/data \
    --lean-project ~/lean/mathlib_v4160 --samples-per-step 32 --min-closers 19 \
    --n-candidates 50000

# Tier 1 (frozen 7B). On a cluster, prefer the sbatch: it enforces node-local Lean staging.
BENCHMARK=fate_m ARM=li INDEX=data/index/li_ft_novel_bm25 \
    sbatch -p gpuA -A <account> -G 1 slurm/prove_benchmark_llm.sbatch
```

**3b. The pass@8 sweep.** Eight seeds are **one array submission** — `SEED` comes from
`SLURM_ARRAY_TASK_ID`, and passing both is refused, because two runs sharing a seed is the same draw
counted twice and inflates every pass@k above it.

```bash
# check the whole configuration WITHOUT a GPU first: index corpus id, benchmark count, model dir, a
# real propose() call, prompt budget, disk, and the projected wall clock against the SLURM limit
python scripts/preflight_sweep.py --benchmark proofnet_test --arm li \
    --index data/index/li_ft_novel_bm25 --data-root <REAL-Prover>/data \
    --model <model-dir> --samples-per-step 32 --premise-free-fraction 0.25 --slurm-time 8

BENCHMARK=proofnet_test ARM=li INDEX=data/index/li_ft_novel_bm25 N_CANDIDATES=50000 \
    SAMPLES=32 PREMISE_FREE_FRACTION=0.25 \
    sbatch --array=0-7 -p gpuA -A <account> -G 1 slurm/prove_benchmark_llm.sbatch
```

Repeat for `ARM=sv INDEX=data/index/sv_ft_novel_lr3e6` (no `N_CANDIDATES` — single-vector has no first
stage and passing one aborts the run) and for `BENCHMARK=fate_m`. Four submissions, 32 jobs,
~124 GPU-h.

**4. Verify, then analyse**

```bash
python scripts/verify_proofs.py --run results/logs/<run_id> \
    --data-root <REAL-Prover>/data --lean-project ~/lean/mathlib_v4160

python scripts/build_table1.py --policy vllm                        # the headline table
python scripts/compare_arms.py --baseline <run> --treatment <run>   # paired significance
python scripts/root_candidate_quality.py --run <run> ...            # T3, needs full traces
python scripts/novel_premise_stratification.py --corpus <corpus> --run <run> ...   # T4
python scripts/verify_arm_distinctness.py --run <run> --run <run>   # T5
python scripts/discordance_profile.py --a <run> --b <run> --control <run>  # T7
python scripts/contamination_audit.py --corpus <corpus> --run <run> ...
```

**pass@8 (T10–T13).** Verify every run first — `passk_union.py` refuses to count one that was never
re-elaborated, and discounts any claim the re-check rejected:

```bash
RUNS="$(ls -1 results/logs | grep '_vllm_.*_<sha>$' | tr '\n' ' ')" \
    sbatch -p multicore slurm/verify_proofs.sbatch

python scripts/passk_union.py --results-root results/exported/logs \
    --benchmark proofnet_test \
    --match search.samples_per_step=32 \
    --match policy_config.premise_free_fraction=0.25 --match n_problems=186
python scripts/passk_profile.py --match search.samples_per_step=32 \
    --match policy_config.premise_free_fraction=0.25
```

**All three `--match` filters are load-bearing.** The budget pilot measured the winning configuration
on ProofNet's first 60 problems at seed 0, so filtering on budget alone drags a 60-problem subset into
an eight-seed estimate. `passk_union` refuses it as a duplicated seed — but that refusal is the only
thing standing between a subset run and the headline table.

If a run finished its problems and died before writing its outcome, it is invisible to every table
(`discover` skips a manifest with no `outcome`) and `--resume` cannot fix it, because with nothing left
to do it returns before finalizing:

```bash
python scripts/finalize_run.py results/logs/<run_id>      # recompute counts from attempts.jsonl
python scripts/repair_attempts.py results/logs/<run_id>   # quarantine genuinely unreadable rows
```

Every run writes a manifest — config, seed, git SHA (with a `-dirty` marker), package versions, Lean
version, hardware, SLURM job id — **before** doing any work, so a job killed by the scheduler still
leaves a record of what it attempted.

**Tests:** `pytest` — 973 hermetic tests plus 10 live-Lean tests that skip unless
`PROOFLENS_LEAN_PROJECT` points at a pre-built Mathlib.

## Layout

```
src/prooflens_prover/
  lean/         backend protocol, LeanInteract backend, proof verdicts
  retrieval/    base, bm25, dense (SV + LI), rank fusion, lean tokenizer
  prover/       best-first search, model-free repertoire, vLLM policy, prompt templates
  data/         benchmark and premise-corpus loaders
  eval/         paired comparison, significance, draws, premise-name resolution
  utils/        seeding, logging, run manifests, io
scripts/        extraction, index building, benchmarks, verification, analysis
slurm/          cluster jobs
tests/          973 hermetic + 10 live-Lean
results/        exported run records and tables
```

## Acknowledgements

Benchmarks and reference numbers from [REAL-Prover](https://github.com/frenzymath/REAL-Prover)
(arXiv:2505.20613) and [FATE](https://github.com/frenzymath/FATE). Lean interaction via
[LeanInteract](https://github.com/augustepoiroux/LeanInteract). Retriever checkpoints from the
predecessor ProofLens premise-selection study.
