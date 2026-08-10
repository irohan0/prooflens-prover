# ProofLens-Prover — retrieval architecture in a live Lean 4 theorem prover

A controlled study of **premise retrieval inside a working Lean 4 proof search**: does the retrieval
architecture change how many theorems a prover can close, and if so, where and why?

Every arm shares one search harness, one premise corpus, one Lean toolchain and one search budget.
Only the retriever varies. Every claimed proof is re-elaborated from the benchmark statement in a
fresh Lean environment before it is counted.

The question is asked twice, with two generators chosen so their weaknesses do not overlap — one is
comparable to published systems but carries prompt and training confounds, the other carries none of
those but sets only a floor:

| tier | generator | why this generator |
|---|---|---|
| **Tier 1** | **REAL-Prover-v1** — 7B, `Qwen2.5-Math-7B` base, stepwise, retrieval-native, held **frozen** | comparable in kind to published systems; retrieval is *substitutive* — premises change the prompt |
| **Track A′** | model-free repertoire — 19 Lean tactics + 5 premise templates, **no language model at all** | no prompt sensitivity, no sampling variance, no training confound; retrieval is *additive* — premises are extra candidates |

**Headline result. Retrieval works. The retrieval architecture does not decide it. Both generators
agree, and they agree for different reasons.**

| | retrieval vs none | late interaction vs single-vector |
|---|---|---|
| **Tier 1** (frozen 7B) | **+14 of 327** pooled, 17.7% → 22.0%, p = 0.0045 | **+0 of 327** — 17 gained, 17 lost, **p = 1.0000** |
| **Track A′** (model-free) | **+19 on FATE-M** (p < 0.0001), **+11 on ProofNet** (p = 0.0010) | −4 (p = 0.34), 0 (p = 1.00), +2 (p = 0.50) |

Late interaction costs **5.8× the index memory and 23–28× the query latency** to draw. Two nulls,
each with a mechanism — and the mechanisms are the contribution:

**Track A′ — the approximation, not the architecture.** Late interaction cannot rank a large corpus
exactly, so it must pre-select candidates with a mean-pooled vector. At the conventional 1,000 that
first stage has recall@10 of **0.443** against exact MaxSim, not the 0.992 an index build reports on
the wrong query distribution. Widening it to 50,000 is worth **+9 theorems on FATE-M and +8 on
ProofNet** with nothing else changed — two thirds of what had looked like a significant
architecture difference.

**Tier 1 — the search reads the part of the distribution retrieval does not improve.** At the root
state, where every arm sees the identical theorem, retrieval measurably improves the generator. But
it improves the *mean* candidate **3–8× more than the best** candidate, and best-first search only
ever consumes the best. Between the two retrieval arms the top candidate is not better at all — it
is very slightly *worse* on both benchmarks. The architecture difference is real, is measurable in
the generator, and sits where the search never looks.

**And the benchmarks barely exercise the regime where late interaction wins.** Its measured strength
is *out-of-distribution* robustness — on novel premises it stays flat where single-vector drops
−17.9%, while single-vector wins in-distribution. But the premises these benchmarks' proofs actually
cite are **2.6× concentrated in the 21.8% of Mathlib the retriever trained on** (fraction unseen
0.43 against a 0.782 base rate). Retrieval runs in-distribution almost all the time here, so the
tie is not evidence against that robustness result — it is what that result predicts once the
deployment distribution is measured.

And the two architectures are not interchangeable even where they tie. Across Tier 1's 327 problems
each retriever solves exactly 72 — but **17 differ in each direction, so either-retriever reaches
89**. The fusion ceiling (+17) is larger than retrieval's own effect (+14). Equal counts, different
theorems.

---

## What this system is — and what it is not

One harness, two generators, and the same `TacticPolicy` protocol behind both — so the search,
benchmarks, manifests, verification and significance machinery are shared, not reimplemented.

| stage | generator | status |
|---|---|---|
| **Track A′** | model-free repertoire (19 tactics + 5 premise templates) | ✅ complete |
| **Tier 1** | REAL-Prover-v1 — 7B, `Qwen2.5-Math-7B` base, stepwise, retrieval-native, frozen | ✅ complete |
| later | Qwen3-4B / Kimina-Prover-Distill-1.7B, LoRA SFT (Track B) | not started |

**Track A′ has no language model.** The policy is a fixed repertoire of 19 Lean tactics (`simp`,
`aesop`, `linarith`, `intro x`, …) plus five templates that consume a retrieved premise (`exact {p}`,
`apply {p}`, `rw [{p}]`, `rw [← {p}]`, `simp [{p}]`).

That is a deliberate design, not an unfinished one. With no generator there is no prompt sensitivity,
no sampling variance and no training confound, so **any difference between arms is attributable to
the retriever alone**. The price is that absolute pass rates are a floor, not a competitive result.

**Tier 1 freezes a real prover and swaps only its retriever.** REAL-Prover-v1 is used because it is
openly released and **already retrieval-native** — trained to consume premises from its own
single-vector retriever, `LeanSearch-PS`. Nothing about the model is touched: same weights, same
sampling parameters transcribed from its shipped config, same prompt template. Only the premises
change.

That cuts both ways and the direction matters: because it was *trained on* `LeanSearch-PS` output,
any other retriever is out-of-distribution for it. The handicap is against our retrievers, so a win
under it would be strong evidence. A null under it is weaker evidence than a null on a
retriever-agnostic model would be, and that is stated in the limitations rather than glossed.

The two tiers also treat retrieval differently, which turns out to matter for reading the results.
For the repertoire, premises arrive as **extra candidates** alongside the fixed tactics — retrieval
can only add. For the LLM, premises **rewrite the prompt** — retrieval changes every candidate, so
it can lose problems the control solved. Track A′ measured zero displacement on both premise-heavy
benchmarks; Tier 1 loses 4–7. That is a property of the policy, not of retrieval, and a study run
only on one of the two would have drawn the wrong general conclusion.

`Qwen2.5-Math-1.5B` was ruled out on a hard constraint rather than a preference: a **4K context
window** cannot hold retrieved premises alongside a proof state.

---

## Results — Tier 1: frozen REAL-Prover-v1 (7B)

**Read the budget before the numbers.** REAL-Prover's published 56.7 / 23.7 are Pass@64×64: 64
passes of 64 nodes × 64 samples, ≈ 4.2M generations per problem. Every row here is a *single* pass
at 64 nodes × 16 samples — 1,024 generations, roughly **1/4,000 of their budget**. Absolute rates are
not comparable to theirs and are not presented as if they were. What *is* controlled, exactly, is the
retriever: identical weights, identical sampling parameters transcribed from their shipped config,
identical prompt template, identical search budget and seed. One variable.

| benchmark | none | ProofLens-SV | ProofLens-LI @50k | Δ retrieval | Δ LI vs SV | SV ∪ LI (oracle) |
|---|--:|--:|--:|--:|--:|--:|
| **FATE-M** (141) | 39 (27.7%) | **46 (32.6%)** | **46 (32.6%)** | **+7** | +0 (11 vs 11) | **57 (40.4%)** [+11] |
| **ProofNet-test** (186) | 19 (10.2%) | **26 (14.0%)** | **26 (14.0%)** | **+7** | +0 (6 vs 6) | **32 (17.2%)** [+6] |
| **pooled** (327) | 58 (17.7%) | **72 (22.0%)** | **72 (22.0%)** | **+14** (p = 0.0045) | **+0** (17 vs 17, **p = 1.0000**) | **89 (27.2%)** [+17] |

`scripts/build_table1.py --policy vllm`. Every claimed proof independently re-elaborated — see
[Verification](#verification).

### T1. The calibration gate, and what it actually showed

The plan's original gate — reproduce REAL-Prover's FATE-M 56.7 within 3% — is unaffordable by a
factor of ~4,000 and would be failed by a perfectly correct harness. It was replaced, before running
anything, by three checks that a correct harness must pass at an affordable budget:

| gate | requirement | measured | |
|---|---|---|---|
| **floor** | the 7B must clear the model-free policy's FATE-M 22.0% | **32.6%** (46 vs 31 problems) | ✅ |
| **direction** | retrieval must help, matching the sign of their 56.7 vs 44.7 ablation | **+14 pooled, p = 0.0045** | ✅ |
| **mechanics** | non-degenerate generation: candidates per expansion in high single digits, near-zero empty and cheat rates, `mean_candidate_logprob` well above −1.5 | **12.1–12.8** distinct candidates of 16 samples; empty rate **0.0001**; cheat rate **0.0**; logprob **−0.81 to −1.00** | ✅ |

One unplanned calibration fell out of it. ReProver — sub-1B, single-vector retrieval, stepwise, one
pass — reports **13.8% on ProofNet**. This harness, retrieval on, reports **14.0%**. That is the
closest like-for-like row available and the agreement is closer than the design deserved; treat it as
evidence the rig is in the right regime, not as a comparison of systems, since the model is 7B rather
than sub-1B.

### T2. Retrieval helps — but only once the benchmarks are pooled

The per-benchmark evidence is split, and pooling is what resolves it. Both benchmarks show the same
+7, but only one is significant on its own:

| contrast | FATE-M | ProofNet-test |
|---|---|---|
| SV vs none | +7, 10 gained / 3 lost, p = 0.0923 — not significant | +7, 8 / 1, **p = 0.0391 — significant** |
| LI vs none | +7, 13 gained / 6 lost, p = 0.1671 — not significant | +7, 8 / 1, **p = 0.0385 — significant** |

FATE-M's failure to reach significance at the same effect size is a statement about power, and it can
be made exactly: **on 19 discordant pairs, exact McNemar cannot reach p < 0.05 for any split closer
than 15–4.** A 13–6 split is not evidence of absence at that sample size. Reporting FATE-M's
non-significance without saying so would have been a false negative dressed as a finding.

Pooled across both benchmarks as one paired fixed-effects test — 327 problems, problem ids namespaced
by benchmark so no pair can span them, one contrast at a time:

| contrast | Δ | gained / lost | 95% CI | p (perm) | p (McNemar) | verdict |
|---|--:|---|---|--:|--:|---|
| SV vs none | **+14** | 18 / 4 | [+0.0153, +0.0703] | **0.0045** | 0.00434 | **SIGNIFICANT** |
| LI vs none | **+14** | 21 / 7 | [+0.0122, +0.0765] | **0.0124** | 0.01254 | **SIGNIFICANT** |
| LI vs SV | **+0** | 17 / 17 | [−0.0367, +0.0336] | **1.0000** | 1.0000 | not significant |

None of the three is heterogeneous — the per-benchmark effects agree in sign in every case, which is
the condition under which a pooled average describes both benchmarks rather than neither. A
**sensitivity analysis excluding harness errors** is computed alongside every figure and changes no
verdict: SV vs none becomes p = 0.0040 on 321 problems, LI vs none p = 0.0135 on 319, LI vs SV
p = 1.0000. The conclusions do not rest on problems that were never really attempted.

`scripts/compare_arms.py --pooled`. The guards are in `eval/compare.compare_pooled`: at least two
pairs, no benchmark repeated, exactly one contrast, and no pair spanning benchmarks. Pooling is
legitimate here because the problem sets are disjoint and the arms are byte-identical across
benchmarks; it is not legitimate as a way to rescue a contrast that has already been tested
per-benchmark and failed, which is why the LI-vs-SV row is reported pooled *and* per-benchmark
(p = 1.00 both times) rather than only in whichever form looks better.

**Retrieval displaces here, and it did not in Track A′.** SV loses 4 problems the control solved and
LI loses 7. For the model-free policy displacement was exactly zero on both premise-heavy benchmarks.
The difference is structural: premises are *extra candidates* for the repertoire and a *prompt
rewrite* for the LLM. Retrieval cannot cost the repertoire a proof; it can and does cost the LLM one.

### T3. Why the architectures tie: the search reads the part retrieval does not improve

A null is worth much more with a mechanism, and the traces already contain one.

The obvious statistic — each run's mean candidate log-probability — orders exactly as the offline
retrieval study predicts, on both benchmarks:

| | none | SV | LI @50k |
|---|--:|--:|--:|
| FATE-M | −0.8865 | −0.8347 | **−0.8097** |
| ProofNet | −1.0042 | −0.9960 | **−0.9827** |

**That table is not a measurement and it is not used as one.** It averages over every expansion of a
run, and the arms diverge after their first tactic — a state the LI arm reached need not exist in the
SV arm's tree at all. Three numbers, three different distributions of proof states. Quoting it as a
comparison would repeat the error described in
[finding 2](#2-premise-used-is-not-premise-needed): a real-looking figure measuring something other
than what it claims.

At **depth 0** there is no divergence. Every arm is prompted with the identical benchmark statement
and differs only in the premises attached. That is a properly paired measurement, and
`scripts/root_candidate_quality.py` computes it from the existing traces — no GPU, no re-run:

| | problems | candidates | mean logprob | best logprob |
|---|--:|--:|--:|--:|
| **FATE-M** | | | | |
| none | 141 | 1568 | −0.9332 | −0.1723 |
| SV | 141 | 1421 | −0.8566 | **−0.1426** |
| LI @50k | 141 | 1457 | **−0.8276** | −0.1528 |
| **ProofNet** | | | | |
| none | 180 | 2139 | −1.1141 | −0.1738 |
| SV | 180 | 2058 | −1.0339 | **−0.1535** |
| LI @50k | 180 | 2073 | **−0.9773** | −0.1578 |

Paired per problem, bootstrap CI (10k) and sign-flip permutation (10k) required to agree:

| contrast | statistic | mean diff | 95% CI | p | verdict |
|---|---|--:|---|--:|---|
| SV vs none | mean | **+0.1023** | [+0.0371, +0.1693] | 0.0034 | **SIGNIFICANT** |
| SV vs none | best | +0.0297 | [+0.0160, +0.0439] | 0.0002 | **SIGNIFICANT** |
| LI vs none | mean | **+0.1064** | [+0.0381, +0.1758] | 0.0031 | **SIGNIFICANT** |
| LI vs none | best | +0.0195 | [+0.0031, +0.0364] | 0.0239 | **SIGNIFICANT** |
| LI vs SV | mean | +0.0041 | [−0.0553, +0.0614] | 0.8886 | not significant |
| LI vs SV | best | **−0.0102** | [−0.0250, +0.0040] | 0.1633 | not significant |

FATE-M above; ProofNet replicates the pattern — SV vs none mean +0.0629 (p = 0.073), LI vs none mean
**+0.1360** (p = 0.0001), LI vs SV mean +0.0701 (p = 0.055, CI touching zero at −0.0002) and best
**−0.0043** (p = 0.615). `results/exported/tables/root_quality_*.json`.

Three things follow.

1. **Retrieval genuinely improves the generator.** This is the mechanism behind the +14 proofs,
   measured at the one state where the comparison is clean.
2. **It improves the mean candidate 3–8× more than the best one** (FATE-M: +0.1064 vs +0.0195 for LI;
   ProofNet: +0.1360 vs +0.0170). Best-first search consumes candidates in rank order, so the
   quantity that decides a proof is the *top* of the distribution — and that is where retrieval's
   effect is smallest. The top candidate already sits at −0.14 to −0.17 per token, about 86%
   per-token probability, with very little headroom left for any retriever to claim.
3. **Between the two architectures the top candidate is not better at all.** LI's edge is confined to
   the mean (+0.0041 on FATE-M, p = 0.89; +0.0701 on ProofNet, p = 0.055 — failing both halves of the
   agreement rule), and on the best candidate it is *negative* on both benchmarks. There is no channel
   through which LI's advantage could become a proof: the search discards precisely the region where
   LI is ahead.

A fourth, independent signal points the same way. The count of **undecodable** completions — tactics
containing a U+FFFD replacement character, which Lean cannot parse — falls monotonically with retrieval
quality on both benchmarks: **96 → 96 → 85** on FATE-M and **257 → 218 → 196** on ProofNet for
none → SV → LI. Better premises produce cleaner generations, by a metric that has nothing to do with
log-probabilities and was collected for an unrelated reason.

So the 17/17 tie is not "the two retrievers are equivalent". It is **the architecture difference is
real, is measurable in the generator, and sits below the resolution of the search** — which is the
claim the predecessor study hypothesised and could not test, having never run a prover.

Two details that would otherwise be confounds, both running against the conclusion rather than for
it. The control produced the *most* root candidates (1568 vs 1421 and 1457 on FATE-M), and a maximum
over more samples is stochastically larger, so the control's best-candidate deficit is if anything
understated; LI also drew slightly more candidates than SV and still lost on best. And the paired sets
are 141/141 on FATE-M against 180/186 on ProofNet, which independently reproduces the known fact that
4 ProofNet statements do not elaborate under Lean v4.16.0.

### T4. Why the predecessor's robustness advantage does not transfer

Late interaction's measured strength in the predecessor study was **out-of-distribution
robustness**: on LeanDojo's `novel_premises` split a matched single-vector retriever dropped −17.9%
from seen to novel premises while LI stayed flat and won in 5/5 seeds — and, importantly, SV won on
the easier in-distribution `random` split. So the two architectures were never uniformly ordered;
which one wins depends on how novel the premises are.

That makes a testable prediction about the tie above. If LI's 17 exclusive wins are the problems
whose proofs need premises the retriever never trained on, and SV's 17 are the problems needing
premises it did, then **+0 is two populations cancelling, not noise** — and the predecessor's claim
survives end to end. `scripts/novel_premise_stratification.py` tests it from the existing logs by
classifying every premise a proof cites against the retriever's own training positives (62,500
premises, extracted from the split it was fine-tuned on).

**It is not two populations.** Pooled across both benchmarks, at 17 exclusive wins per arm:

| | problems | premises cited/proof | fraction unseen |
|---|--:|--:|--:|
| only SV | 17 | 11.7 | 0.418 |
| only LI @50k | 17 | 7.3 | **0.449** |

Difference **+0.0310**, permutation p = **0.7024**. Excluding tokens that are tactic syntax rather
than citations moves it to +0.0419, p = 0.7151 — same answer. There is no enrichment to find.

**The reason is in the levels, not the difference.** Every arm's proofs sit at a fraction-unseen of
**0.43 against a base rate of 0.782** — the probability that an arbitrary corpus premise is one the
retriever never trained on:

| | none | SV | LI @50k | base rate |
|---|--:|--:|--:|--:|
| fraction of cited premises unseen | 0.426 | 0.427 | 0.434 | **0.782** |
| enrichment toward the training distribution | 2.64× | 2.63× | 2.60× | 1× |

The premises these benchmarks actually need are **2.6× concentrated** in the 21.8% of Mathlib the
retriever trained on (3.2× under the tactic-syntax sensitivity). That is not a coincidence and it is
not a flaw in the benchmarks: the retriever's training positives are the premises human Mathlib
proofs cite, and benchmark proofs cite the same commonly-used lemmas. Retrieval here is running
**in-distribution almost all of the time.**

So the end-to-end tie is not evidence against the predecessor's finding — it is what that finding
predicts once the deployment distribution is measured. LI's advantage is on novel premises; SV's is
on familiar ones; these benchmarks are overwhelmingly familiar, so the regime where late interaction
wins is barely exercised. **The practical statement is conditional: choose multi-vector retrieval
when the premise distribution is genuinely novel relative to training, and standard Lean benchmarks
are not that.**

`results/exported/tables/novel_premise_*.json`.

### T5. The tie is not one run counted twice

Equal counts on two independent benchmarks — 46 vs 46 and 26 vs 26 — is exactly the shape a
duplicated or mislabelled run would take. If one arm had been run twice and one copy relabelled, the
two "arms" would agree exactly and every figure derived from them would be an artefact. That
deserves a test rather than an assurance, and `scripts/verify_arm_distinctness.py` is it.

The decisive signal is proof text. At a fixed `--seed` the vLLM engine is deterministic, so two runs
of the *same* arm agree on every shared proof, character for character.

| SV vs LI @50k | FATE-M | ProofNet-test | a duplicate would give |
|---|--:|--:|--:|
| both solved | 35 | 20 | — |
| **byte-identical proofs** | **22.9%** | **45.0%** | **~100%** |
| discordant problems | 22 | 12 | **0** |
| recorded retriever | `sv` / `li` | `sv` / `li` | identical |
| retrieval latency | 37.6 / 929.6 ms (24.7×) | 36.4 / 1010.6 ms (27.7×) | identical |

Four independent signals, and a duplicate fails all four at once. **Verdict: distinct runs.**

**And the tie is unremarkable.** Given `k` problems where two equivalent arms disagree, each falling
either way with probability ½, an exact tie has probability `C(k, k/2) / 2ᵏ` — **0.168** at FATE-M's
22 disagreements and **0.226** at ProofNet's 12. Those are the *modal* outcomes: an exact tie is the
single most likely result when two arms are equivalent, more likely than any other delta. Both ties
together carry probability 0.038, which is an ordinary coincidence rather than a red flag.

So the equality of *totals* is the least informative thing in the table. What matters is that the
solved *sets* differ substantially — which is [T6](#t6-equal-counts-different-theorems--and-the-ceiling-exceeds-the-effect),
and whether that difference exceeds sampling noise is
[Phase 2](#h8--the-architecture-null-is-stable-under-re-sampling).

`results/exported/tables/arm_distinctness_*.json`.

### T6. Equal counts, different theorems — and the ceiling exceeds the effect

| benchmark | SV | LI @50k | LI-only | SV-only | SV ∪ LI |
|---|--:|--:|--:|--:|--:|
| FATE-M | 46/141 | 46/141 | 11 | 11 | **57 (40.4%)** |
| ProofNet-test | 26/186 | 26/186 | 6 | 6 | **32 (17.2%)** |
| pooled | 72/327 | 72/327 | 17 | 17 | **89 (27.2%)** |

Perfect symmetry at every level — 11/11, 6/6, 17/17 — which is what a fair coin does, and is the
strongest form the null can take. But **34 of the 89 problems in the union are solved by exactly one
arm**, and the union sits **+17 above either arm while retrieval's whole measured effect is +14**.
Combining the two architectures has more headroom than adding retrieval had in the first place.

The union is an oracle: it picks the winning retriever per problem, knowing the answer. It is a
ceiling, not a result — but it is the ceiling a fusion arm chases, and it needs no new retriever and
no training. See [H4](#h4--the-two-architectures-are-complementary-so-fusion-should-beat-both).

### T7. Where the searches actually die

Every problem-arm's terminal status, from the run records:

| | proved | `no_candidates` | `max_expansions` | wall clock | error |
|---|--:|--:|--:|--:|--:|
| FATE-M / none | 39 (28%) | 46 (33%) | 56 (40%) | 0 | 0 |
| FATE-M / SV | 46 (33%) | 44 (31%) | 51 (36%) | 0 | 0 |
| FATE-M / LI | 46 (33%) | 41 (29%) | 53 (38%) | 0 | 1 |
| ProofNet / none | 19 (10%) | 86 (46%) | 76 (41%) | 0 | 5 |
| ProofNet / SV | 26 (14%) | 77 (41%) | 77 (41%) | 1 | 5 |
| ProofNet / LI | 26 (14%) | 77 (41%) | 76 (41%) | 1 | 6 |

Two things fall out of this, and both matter.

**The wall clock binds once in 981 problem-arms.** Every `exhausted` search hit `max_expansions`
instead. Whatever is limiting this prover, it is not time per problem — which is the sharpest form of
the [H3](#h3--the-generator-not-the-search-budget-is-the-binding-constraint) claim, and it is measured
rather than argued.

**Between 29% and 46% of problems end in `no_candidates`** — the search stops because the generator
has nothing usable left to propose, not because it ran out of budget. That is the largest single
failure mode on ProofNet, larger than the expansion cap. And **retrieval shrinks it**: 46 → 44 → 41 on
FATE-M and 86 → 77 → 77 on ProofNet, with the LI arm lowest on FATE-M. Retrieval's contribution is
partly to give the model something to say at states where it would otherwise fall silent, which is the
same story the root-state log-probabilities tell from the other direction.

### T8. Cost

LI's accuracy gap against SV is zero and its cost gap is not.

| | wall clock | retrieval queries | ms/query | time in retrieval |
|---|--:|--:|--:|--:|
| FATE-M / none | 0.82 h | 0 | — | — |
| FATE-M / SV | 0.98 h | 3,874 | **37.6** | 0.04 h (4%) |
| FATE-M / LI @50k | **2.17 h** | 4,137 | **929.6** | **1.07 h (49%)** |
| ProofNet / none | 1.71 h | 0 | — | — |
| ProofNet / SV | 2.04 h | 5,628 | **36.4** | 0.06 h (3%) |
| ProofNet / LI @50k | **3.65 h** | 5,658 | **1,010.6** | **1.59 h (44%)** |

**25–28× the query latency and 1.8–2.2× the wall clock**, for 17 problems gained and 17 lost. The
sharpest way to put it: LI spends **roughly half of its entire run inside the retriever** — 1.07 h of
2.17 h, 1.59 h of 3.65 h — where SV spends 3–4%. Half the compute of the more expensive arm buys
nothing measurable.

Both arms returned a mean of exactly 10.0 premises per query, so the prompt budget was identical and
the difference is ranking alone. No prompt was ever truncated in any run (`max_prompt_tokens_seen`
1,536 against a 3,840-token limit), which removes truncation as a possible confound between arms.

On this corpus, at this scale, with either generator, **multi-vector retrieval does not pay for
itself.**

---

## Results — Track A′: model-free policy

Retrieval arms over the model-free repertoire. With no generator in the loop, any difference between
arms is the retriever alone. `@1k` / `@50k` is the LI arm's **first-stage candidate budget** — see
[finding 4](#4-the-approximation-was-the-effect-not-the-architecture), which is the main result of
this half.

| benchmark | none (floor) | ProofLens-SV | ProofLens-LI @50k | Δ LI vs none | Δ LI vs SV | SV ∪ LI (oracle) |
|---|--:|--:|--:|--:|--:|--:|
| **FATE-M** (141) | 12 (8.5%) | **35 (24.8%)** | 31 (22.0%) | **+19** (p<0.0001) | −4 (p=0.34) | **38 (27.0%)** [+3] |
| **ProofNet-test** (186) | 9 (4.8%) | **20 (10.8%)** | **20 (10.8%)** | **+11** (p=0.0010) | 0 (p=1.00) | **24 (12.9%)** [+4] |
| miniF2F-test (244) | 78 (32.0%) | 77 (31.6%) | 79 (32.4%) | +1 (p=1.00) | +2 (p=0.50) | 79 (32.4%) [+0] |

Significance requires the paired bootstrap CI (10k) and the sign-flip permutation test (10k) to
**agree**; the exact McNemar p is reported since the outcome is binary and paired.

**SV ∪ LI** is the oracle union — problems solved by *either* retriever, with the gain over the
better single arm in brackets. It is not achievable by one retriever and it is not a fusion result;
it is the ceiling a fusion arm could reach. See
[finding 5](#5-equal-counts-different-theorems).

Every claimed proof in every row is independently re-elaborated. See
[Verification](#verification).

### 1. Retrieval helps, and only where premises are needed

Against the no-retrieval control, LI gains **+19 problems on FATE-M** (12/141 → 31/141, 95% CI
[+0.078, +0.192], permutation p = 0.0002, McNemar p < 0.00001) and **+11 on ProofNet** (9/186 →
20/186, CI [+0.027, +0.097], p = 0.0011, McNemar p = 0.00098). On both, the control's solved set is a
strict subset of LI's: **zero displacement**, retrieval never cost a problem.

The effect tracks whether the benchmark *requires* citing lemmas:

| benchmark | domain | Δ vs none | verdict |
|---|---|--:|---|
| FATE-M | graduate abstract algebra | **+19** | significant |
| ProofNet-test | undergraduate, mixed | **+11** | significant |
| miniF2F-test | competition arithmetic | +1 | not significant |

miniF2F is close to the strongest form of a null: one problem gained out of 244, and at the
1,000-candidate budget the solved sets were **identical**. Its problems fall to `linarith` /
`nlinarith` / `omega` / `norm_num`, where retrieval has no leverage — and where offering premises can
cost a problem rather than win one (SV: 77 against the control's 78).

### 2. "Premise used" is not "premise needed"

Two distinct quantities, routinely conflated:

* **used** — of all proofs found, how many name a retrieved premise.
* **needed** — of the proofs the *control could not find*, how many name a premise.

| benchmark | used | needed |
|---|--:|--:|
| FATE-M | 26/31 = 84% | **19/19 = 100%** |
| ProofNet-test | 17/20 = 85% | **11/11 = 100%** |
| miniF2F-test | 18/79 = 23% | 1/1 |

Only the second is causal, and on both premise-heavy benchmarks it is **100%**: every problem
retrieval won was won by naming a premise. Five FATE-M proofs used a premise (`exact conj_pow`,
`rw [orderOf_inv]`, `apply Fintype.card_pi_const`, …) for problems the control also solved with a
bare `simp`/`aesop` — the premise tactic merely ranked first. On miniF2F the gap is near-total: 23%
of proofs cite a premise, and it changed the outcome for one problem in 244.

Distinguishing them requires a paired control. Reading proof text cannot do it.

### 3. Multi-vector vs single-vector: no measurable difference

Three benchmarks, three nulls:

| benchmark | SV | LI @50k | Δ | 95% CI | p (McNemar) |
|---|--:|--:|--:|---|--:|
| FATE-M | 35/141 | 31/141 | −4 | [−0.071, +0.014] | 0.34 |
| ProofNet-test | 20/186 | 20/186 | 0 | [−0.032, +0.032] | 1.00 |
| miniF2F-test | 77/244 | 79/244 | +2 | [+0.000, +0.021] | 0.50 |

On FATE-M the 10 discordant pairs split 7–3 against LI; on ProofNet they split 4–4. Neither is a
signal — 7–3 is what a fair coin does.

So the study's central question gets a null answer at this scale: **with the generator held fixed,
late interaction does not close more theorems than a matched single-vector retriever.** It does not
close fewer either.

An earlier version of this table reported **−13 (p = 0.0023)** for the same comparison on FATE-M, and
that number was wrong in a specific and instructive way. It is the subject of the next section, which
is the main result of this work.

### 4. The approximation was the effect, not the architecture

Late interaction cannot rank a large corpus exactly, because its defining property is keeping **one
vector per token** instead of one per premise.

| | vectors stored | index size | exact search |
|---|--:|--:|---|
| ProofLens-SV | 276,070 (one per premise) | 943 MB | one matrix-vector product — **feasible** |
| ProofLens-LI | **21,752,080** (78.8 per premise) | 5.5 GB fp16 | see below — **infeasible** |

Exact full-corpus MaxSim requires scoring every query token against every premise token: a
`384 × 21,752,080` score matrix, ≈ **8.4 billion floats (33 GB) for a single query**. At roughly
5,300 queries per benchmark run that is not an engineering inconvenience, it is a different
algorithm.

So LI is necessarily **two-stage**: mean-pooled vectors select the top `n_candidates`, and MaxSim
only ever sees those. A premise the first stage drops cannot be recovered by late interaction, no
matter how good the late interaction is. SV has no such stage; it ranks all 276,070 exactly.

The `recall@10 = 0.992` printed at index-build time did **not** license ignoring this. It was
measured with *premise embeddings* as probes, so it describes a premise retrieving its neighbours —
not a proof state retrieving a premise. Wrong distribution.

**Measured on 141 real FATE-M queries against exact full-corpus MaxSim:**

| n_candidates | % of corpus | recall@10 | queries lossless |
|--:|--:|--:|--:|
| **1,000** (the original default) | 0.36% | **0.443** | 9/141 |
| 5,000 | 1.81% | 0.696 | 32/141 |
| 20,000 | 7.24% | 0.888 | 81/141 |
| **50,000** (used above) | 18.11% | **0.979** | 124/141 |

`results/exported/tables/li_recall_fate_m.json`. The index-build figure overstated recall by **2.2×** purely
by probing with the wrong distribution.

At 1,000 the LI arm was ranking a candidate set missing more than half of its own true top-10, and
fewer than 1 query in 15 got a complete one. SV, ranking all 276,070 exactly, was not handicapped at
all. So the two arms were never comparable.

**Widening the first stage from 1,000 to 50,000 is worth +9 problems on FATE-M and +8 on ProofNet —
and nothing else changed.**

| LI arm | proved | recall@10 | ms/query |
|---|--:|--:|--:|
| FATE-M, `n_candidates` = 1,000 | 22/141 (15.6%) | 0.443 | 79.1 |
| FATE-M, `n_candidates` = 50,000 | **31/141 (22.0%)** | 0.979 | **1030.9** |
| ProofNet, `n_candidates` = 1,000 | 12/186 (6.5%) | — | 76.9 |
| ProofNet, `n_candidates` = 50,000 | **20/186 (10.8%)** | — | **1128.0** |

On FATE-M: 95% CI [+0.028, +0.106], permutation p = 0.0028, exact McNemar p = 0.0039 —
**significant**, with zero displacement (every problem solved at 1,000 is still solved at 50,000).
Same encoder, same index, same policy, same search budget, same seed; one integer differs. Unchanged
when the single harness error is excluded. `results/exported/tables/h1_li_1k_vs_50k.json`.

ProofNet replicates it independently at +8, on a different problem distribution — its recall curve
was never measured, so the size of the effect there was a prediction, not a fit.

That is the finding. **Two thirds of what looked like an architecture difference was the
approximation the architecture forces you into** — 9 of the 13 problems. The remaining 4 are noise
(p = 0.34). A late-interaction result reported without stating its first-stage recall on the real
query distribution is not interpretable, and the plausible-looking 0.992 from index-build time is
exactly the kind of number that makes it feel interpretable.

### 5. Equal counts, different theorems

The null in [finding 3](#3-multi-vector-vs-single-vector-no-measurable-difference) says the two
architectures solve the *same number* of problems. It does not say they solve the *same problems*, and
on the evidence they do not.

| benchmark | SV | LI @50k | LI-only | SV-only | SV ∪ LI |
|---|--:|--:|--:|--:|--:|
| FATE-M | 35/141 | 31/141 | 3 | 7 | **38 (27.0%)** |
| ProofNet-test | 20/186 | 20/186 | 4 | 4 | **24 (12.9%)** |
| miniF2F-test | 77/244 | 79/244 | 2 | 0 | 79 (32.4%) |

ProofNet is the clean case: an exact tie at 20 each, with **four problems in each direction**. A
single number reported for either arm hides that completely. The union is above both arms on both
premise-heavy benchmarks — +3 and +4 — which is only possible because the disagreement is real.

The union is an **oracle**: it picks the winning retriever per problem, knowing the answer. It is not
a result, it is a *ceiling* — the most a fusion arm (reciprocal-rank fusion over both rankings, or
simply concatenating both top-k lists) could reach. But it is a ceiling worth chasing precisely
because the two arms tie: a fusion arm is the one configuration this evidence says should beat both,
and it needs no new retriever, no training, and one run per benchmark.

### Cost

The accuracy gap closes; the cost gap does not. LI needs **23–27× SV's query latency** to reach a
result that is statistically indistinguishable from it.

| | size | ms/query | ranks |
|---|--:|--:|---|
| BM25 | 169 MB | **9.8** | all 276,070 exactly |
| ProofLens-SV (768-dim) | 943 MB | **41.8–44.7** | all 276,070 exactly |
| ProofLens-LI @1k (fp16, 21.7M token vectors, 78.8/premise) | ~5.5 GB | 76.9–85.9 | 0.36% of corpus, recall 0.443 |
| ProofLens-LI @50k | ~5.5 GB | **1030.9–1128.0** | 18.1% of corpus, recall 0.979 |

Latency scales with the rerank, not the index: 50× the candidates costs 13× the wall clock (the
pooled first stage and index load are fixed overheads). On this corpus, **the practical case for
late interaction is weak** — 5.8× the memory and 23–28× the latency for no measurable accuracy gain.

That was written as a conclusion about a 276k-premise corpus and a model-free policy, with the caveat
that a generator able to exploit finer-grained premise ranking might change it. **Tier 1 tested that
and it did not change it** — a frozen, retrieval-native 7B shows the same null at the same cost ratio,
and [T3](#t3-why-the-architectures-tie-the-search-reads-the-part-retrieval-does-not-improve) explains
why: the generator *does* respond to the better ranking, in the part of its candidate distribution the
search throws away.

| | |
|---|--:|
| Premise corpus | 276,070 premises (Mathlib v4.16.0) |
| `import Mathlib`, NFS | 439–691 s |
| `import Mathlib`, node-local | **158 s** |
| FATE-M run, LI @50k | 4.05 h wall clock (5,330 retrieval calls) |

For context on scale: REAL-Prover's own retriever, `LeanSearch-PS`, is built on
`intfloat/e5-mistral-7b-instruct` — a **7B single-vector** embedding model. Both retrievers here are
ModernBERT-base class (~149M), roughly 47× smaller, which is worth stating before comparing any
absolute retrieval quality to their published numbers.

LI at `query_length=256` measured 77.9 ms; raising it to the locked 384 cost ~10% latency and
changed which premises were retrieved (5,288 → 5,330 retrieval calls) but left the solved set
unchanged — the problems a model-free policy closes have short goals, where truncation never bit.

### Verification

`scripts/verify_proofs.py` re-elaborates every claimed proof from the benchmark statement in a fresh
Lean environment, sharing nothing with the search but the cheat-token regex. `sorry`, `admit` and
`native_decide` are rejected before execution and again at verification.

**Tier 1** — all six runs, eight verification jobs, every one reporting `ALL RUNS VERIFIED CLEAN`:

| run | verified |
|---|---|
| FATE-M / none · SV · LI @50k | **✅ all clean** |
| ProofNet / none · SV | **✅ all clean** |
| ProofNet / LI @50k | **26/26 ✅** |

**Track A′:**

| run | verified |
|---|---|
| FATE-M / SV | **35/35 ✅** |
| FATE-M / LI @50k | **31/31 ✅** |
| FATE-M / LI @1k | 22/22 ✅ |
| FATE-M / none | ✅ |
| ProofNet / SV | **20/20 ✅** |
| ProofNet / LI @50k | **20/20 ✅** |
| ProofNet / LI @1k | ✅ |
| miniF2F / SV | ✅ |
| miniF2F / LI @50k | ✅ |
| miniF2F / LI @1k | 78/78 ✅ |

Every reported proof re-elaborates cleanly. The results table is not provisional.

### Run records

`results/exported/` holds **28 runs** — each one's manifest and per-problem outcomes, with the
per-tactic traces stripped (they are the bulk of the file, and every number in both tables above is
recomputable without them). Both tables regenerate from these records alone:

```bash
python scripts/build_table1.py --results-root results/exported/logs --policy vllm
python scripts/build_table1.py --results-root results/exported/logs --policy repertoire
```

The one exception is [T3](#t3-why-the-architectures-tie-the-search-reads-the-part-retrieval-does-not-improve),
which reads the traces and therefore needs the full logs; its computed output ships as
`results/exported/tables/root_quality_*.json`.

Superseded runs are kept rather than deleted, because several of them are the evidence for defects
described here. `build_table1.py` selects the most recent *finalised* run per (benchmark, arm), so
their presence does not affect either table. FATE-M / LI is the most-revised lineage:

| run | note |
|---|---|
| `..._5fcb0b4` | 5-problem smoke run |
| `..._027cff0` | superseded: a REPL restart voided problems 109–140 |
| `..._a59423a` | superseded: `query_length=256` |
| `..._9cc3513` | superseded: `query_length=384`, `n_candidates` unrecorded |
| `..._42e2fb4` | retained: `n_candidates=1000`, the baseline for finding 4 |
| `..._5a6054c` | **current (Track A′)**: `n_candidates=50000` |
| `..._174ee8e`, `..._9f89a43` | Tier 1 smoke runs — `174ee8e` is the 0/5 under the wrong chat template |
| `..._8175a00`, `..._badc9e4` | **current (Tier 1)** |

The three Tier 1 smoke runs are deliberately retained: the 0/5 under the deepseek template against
2/5 under ChatML, on the same five problems with the same weights, is the whole evidence for the
prompt-format finding.

---

## Conclusions

1. **Runtime premise retrieval measurably improves a live Lean prover, and the result holds across
   two generators that share nothing but the harness.** Frozen 7B: +14 of 327 pooled (17.7% → 22.0%,
   p = 0.0045). Model-free: +19 on FATE-M (p < 0.0001) and +11 on ProofNet (p = 0.0010). It holds only
   on benchmarks whose proofs require citing lemmas the policy cannot otherwise reach — on competition
   arithmetic the effect is one problem in 244. Benchmark choice determines whether a retrieval claim
   is measurable at all.
2. **For the retrieval *architecture*, the answer is a null, under both generators and on every
   benchmark.** Frozen 7B: +0 of 327, with 17 problems gained and 17 lost, p = 1.0000. Model-free: −4
   (p = 0.34), 0 (p = 1.00), +2 (p = 0.50). Late interaction needs 5.8× the index memory and 23–28×
   the query latency to draw. **Multi-vector retrieval does not pay for itself at this scale**, and the
   obvious escape route — "the model-free policy is too weak to exploit better ranking" — was tested
   directly and closed.
3. **The null has two distinct mechanisms, and they are the contribution.** For the model-free arm the
   apparent difference was the **two-stage approximation, not the architecture**: LI's first-stage
   recall@10 on real queries is 0.443 at the conventional 1,000-candidate budget and 0.979 at 50,000,
   and closing that gap is worth +9 on FATE-M (p = 0.0039) and +8 on ProofNet with nothing else
   changed — two thirds of a significant "single-vector wins" result was an artefact of a default. For
   the 7B arm the mechanism is **resolution**: at the root state retrieval improves the mean candidate
   3–8× more than the best candidate, and best-first search consumes only the best. The architecture
   difference is real and measurable in the generator; it sits below the resolution of the search.
4. **A null on the counts is not a null on the theorems, and the gap is large.** Under the frozen 7B
   each retriever solves exactly 72 of 327 while **34 problems are solved by exactly one of them** —
   the union reaches 89. The fusion ceiling (+17) exceeds retrieval's entire measured effect (+14).
   Rank fusion is the one configuration this evidence predicts should beat both arms, and it needs no
   new retriever and no training.
5. **Retrieval attribution needs a paired control, and for a language model it needs more than that.**
   The intuitive metric — how many proofs cite a retrieved premise — overstated the causal contribution
   by 60% on FATE-M; against a paired control the causal figure is 100% on both premise-heavy
   benchmarks. Under the LLM the same metric becomes **unmeasurable and is withheld**: it is decidable
   from proof text only when every tactic is either a fixed closer or a premise template, so applied to
   generated tactics it would report 100% regardless of what retrieval contributed. Reporting it there
   would have produced a confident, meaningless number.
6. **Any late-interaction result should state its first-stage recall on the real query distribution.**
   The plausible-looking 0.992 available at index-build time is measured with premise embeddings as
   probes — a premise retrieving its neighbours, not a proof state retrieving a premise — and
   overstates recall by 2.2×. Without that number a late-interaction comparison is not interpretable.
7. **Whether retrieval can *cost* a proof is a property of the policy, not of retrieval.** For the
   repertoire, premises are extra candidates and displacement was exactly zero on both premise-heavy
   benchmarks. For the LLM, premises rewrite the prompt, and retrieval lost 4–7 problems the control
   had solved. A study run on only one of the two would have generalised the wrong way.
8. **A model-free policy reaches 8.5–32% depending on benchmark**, which sets the floor any generator
   must clear to demonstrate it is contributing. The frozen 7B clears it — 32.6% against 22.0% on
   FATE-M — but by less than the 4,000× budget gap between this harness and its published
   configuration would suggest, which is itself a useful calibration on how much of a published
   stepwise-prover number is search budget rather than model.

## Limitations

**Tier 1 (frozen 7B):**

* **The search budget is ~1/4,000 of the published configuration** — one pass at 64 nodes × 16
  samples, 1,024 generations per problem, against REAL-Prover's Pass@64×64 at ≈4.2M. Absolute rates
  are not comparable to their 56.7 / 23.7. What the budget does *not* affect is the contrast: every
  arm has the identical budget.
* **The model was trained on its own retriever's output**, so both ProofLens retrievers are
  out-of-distribution for it. This biases against them, which makes the retrieval-vs-none result
  conservative — but it also **weakens the architecture null**: LI might be handicapped by
  distribution shift rather than by architecture. A retriever-agnostic generator (Track B) is the
  clean test and has not been run.
* **The null is a null at one budget.** Both retrieval arms are within one problem of each other on
  every benchmark, but a 16-sample single pass reads only the top of each candidate distribution.
  T3 shows LI's advantage lives in the mean rather than the max, so a *wider* search — more samples
  per state, or sampling deeper into the ranking — is exactly the regime where the null might break.
  That is a prediction the design makes and does not test.
* **Retrieval is substitutive here**, so displacement is expected and observed (4–7 problems lost).
  The pooled Δ is a net figure; the gained/lost split is reported alongside it in every row rather
  than being summarised away.
* Two Tier 1 arms were run per benchmark from a single job each, so there is **no seed replication**:
  the LLM samples at temperature 1.5 and a re-run would not reproduce the solved set exactly. The
  17/17 symmetry is one sample of a paired difference, not an average over seeds.
* **The `none` arm's manifest records `index: data/index/li_ft_novel_bm25`**, which it never opened.
  The sbatch passes its default `INDEX` regardless of arm and the manifest records the argument, not
  the use; `policy_config.retriever: none` and `n_queries: 0` are the fields that say what actually
  happened. Cosmetic, but it reads like a mislabelled run until those two are checked.
* **The unseen-premise metric has three known weaknesses**, all reported rather than corrected.
  Premise names are matched across two Mathlib versions (the predecessor traced commit `29dcec07`,
  this project indexes v4.16.0), and **2,331 of 62,500 training premise names — 3.7% — have no
  v4.16.0 equivalent**, so a renamed premise is misclassified as unseen. A citation is resolved by
  name against the corpus, and an abbreviated name can denote several premises; the tie is broken
  toward *seen*, which biases against the finding the analysis was looking for. And whether tactic
  keywords count as citations changes the fraction materially (0.43 → 0.27); both settings are
  reported and both give the same verdict on the contrast.
* The unseen-premise contrast has 17 problems per arm even pooled. It can exclude a large enrichment,
  not a small one; the levels it measures (2.6× in-distribution concentration) rest on all 130 solved
  problems and are much better determined than the difference.
* The local premise corpus used for that analysis has **276,108 names against the 276,070 the index
  asserts**, a 0.014% difference that does not affect any reported figure but means the file is not
  byte-identical to the one the arms were indexed over.
* The depth-0 analysis paired 180 of 186 ProofNet problems. Four are the statements that do not
  elaborate under Lean v4.16.0 (`277`, `336`, `340`, `342`, with the elaboration errors recorded in the
  run files); two more produced no root expansion at all, and the exported records cannot say why
  because diagnosing it needs the per-tactic traces that are stripped from them. Separately, 4 problems
  errored *mid-search* with `Unknown proof state` after the root had been expanded, so they appear in
  the depth-0 analysis but not in the sensitivity denominators — 8 distinct ProofNet problems failed in
  one arm or another, which is why the sensitivity `n` is 319–321 rather than a single number.

**Track A′ (model-free):**

* The policy is model-free; absolute rates are **not** comparable to 7B-class systems
  (REAL-Prover-v1: 56.7 FATE-M / 23.7 ProofNet; ReProver: 13.8 ProofNet).
* **On miniF2F, SV solves 77 against the control's 78** — retrieval cost a problem there. Premise
  tactics compete for candidate slots with the shared repertoire, and `--min-closers 19` reduces
  that displacement without eliminating it. It is not significant at n = 244, but it is the right
  sign for the mechanism and the reason the paired `sv vs none` comparison is now reported.
* 4 ProofNet statements do not elaborate under Lean v4.16.0; the effective denominator is 182. 186
  is reported for comparability with published numbers.
* The oracle union is an upper bound obtained by knowing which arm wins each problem. A real fusion
  arm has not been run, and would land somewhere below it.
* LI's recall curve was measured on FATE-M only. The +8 on ProofNet is consistent with the same
  mechanism but its first-stage recall there is unmeasured.
* `AutoLeanServer` restarts the REPL on system-wide memory pressure, which on a shared cluster can
  be triggered by other jobs. Recoveries are counted in each manifest (`n_stale_env_recoveries`).
* Retrieval queries the *statement* at the root and the pretty-printed goal thereafter; no proof
  context beyond the goal is used.
* The recall measurement encodes benchmark *statements*, not mid-search proof states — same
  distribution as the root states, where most retrieval calls happen, but not identical to all of
  them. It needs no Lean server, which keeps it a pure-retrieval measurement.

---

## Next steps, as hypotheses

Each step is stated as a claim that the experiment can falsify, with what each outcome would mean.

The remaining work is sequenced into phases, cheapest and most decisive first. A phase is only
started once its predecessor has landed, so the write-up is always complete at the last finished
phase rather than half-finished across several.

| phase | question | cost | status |
|---|---|---|---|
| **1** | Does LI win the problems needing premises it never trained on? ([H7](#h7--the-null-is-two-populations-cancelling-rather-than-noise--answered)) | analysis only, no GPU | ✅ **answered — no** |
| **2** | Does the +0 survive re-sampling — and is the disagreement above noise? ([H8](#h8--the-architecture-null-is-stable-under-re-sampling)) | ~17.7 GPU-h | ⏳ code ready, needs 8 cluster runs |
| **3** | Does LI pull ahead at wider search? ([H5](#h5--the-null-is-a-property-of-the-search-width-not-of-the-retrievers)) | ~16 GPU-h | not started |
| **4** | Does the benchmark-dependence hold for an LLM? (miniF2F under Tier 1) | ~9 GPU-h | not started |
| **5** | Does fusing the two rankings beat both? ([H4](#h4--the-two-architectures-are-complementary-so-fusion-should-beat-both)) | ½ day + 8 GPU-h | not started |

Phase 2 is placed before the more interesting phases deliberately. The central result is a **null**,
and the one thing a null cannot survive without is a measure of its own variance — see
[H8](#h8--the-architecture-null-is-stable-under-re-sampling).

Deferred rather than sequenced: **Track B** ([H6](#h6--a-retriever-agnostic-generator-removes-the-last-confound))
and PutnamBench. Both are in the original plan and both are now poor value. Track B adds the training
confound the frozen-model design exists to avoid; PutnamBench cannot discriminate between retrievers
at this scale — the plan conceded that before any of this ran, and a 7B at 1/4,000 budget would land
near ReProver's zero.

### H1 — LI lost to its candidate generator, not to late interaction ✅ ANSWERED

> **Claim.** LI's two-stage retrieval drops relevant premises before MaxSim ever sees them. At a
> first-stage budget large enough to recover them, LI's FATE-M score rises materially above 22/141.

**Confirmed, and it changed the chapter's headline.** The three outcomes were written down before
running anything:

| outcome | reading | status |
|---|---|---|
| recall@1,000 ≈ 0.99 | the approximation is not the problem; SV beats LI on merit | ❌ ruled out (recall is 0.443) |
| recall@1,000 ≪ 0.99, LI improves when re-run | the −13 measured **candidate generation**, not late interaction | ✅ **this one** — 22 → 31, p = 0.0039 |
| recall low but LI does **not** improve | late interaction cannot exploit the premises it recovers | ❌ ruled out |

The −13 (p = 0.0023) that this README previously reported as its central result was **two thirds
artefact**. At 0.979 recall the gap is −4 and not significant (p = 0.34). What survives is a null on
the architecture plus a positive, significant result about the approximation — see
[finding 4](#4-the-approximation-was-the-effect-not-the-architecture). ProofNet then replicated it at
+8 on a distribution whose recall curve was never measured, so the effect size there was a prediction
rather than a fit.

**What is still open.** LI at 50,000 candidates is not exact: recall 0.979, and 17 of 141 queries
still see an incomplete top-10. Exact full-corpus MaxSim would cost ~26 s/query (≈38 h for one
benchmark run), so the true architecture ceiling remains unmeasured; the −4 is a lower bound on LI's
best possible showing, and it is already inside the noise.

### H2 — the benchmark, not the retriever, determines whether retrieval can matter ✅ SUPPORTED

> **Claim.** Retrieval's measurable effect is a function of how much a benchmark's proofs depend on
> citing lemmas, and is near zero where tactic automation suffices.

All nine cells are now filled at the corrected budget: FATE-M **+19** (p < 0.0001), ProofNet **+11**
(p = 0.0010), miniF2F **+1** (p = 1.00). The ordering matches problem domain exactly — graduate
abstract algebra, undergraduate mixed, competition arithmetic — and on miniF2F retrieval can even cost
a problem (SV 77 vs the control's 78).

The mechanism check agrees: premise-*needed* is 100% on FATE-M and ProofNet, and 1 problem in 244 on
miniF2F, where 23% of proofs cite a premise without needing to.

### H3 — the generator, not the search budget, is the binding constraint

> **Claim.** The model-free policy is limited by the tactics it can propose, not by how long it may
> search. Adding a language model moves the numbers; adding search budget does not.

**Confirmed, and it did not rescue late interaction.** Two pieces of evidence. First, on the
model-free policy **51 of 71** exhausted FATE-M searches hit `max_expansions` rather than the wall
clock — the search ran out of ideas, not time. Second, replacing the policy with a 7B generator moved
FATE-M from 31/141 to 46/141 and ProofNet from 20/186 to 26/186 at the *same* search budget. The
generator was indeed the binding constraint.

But the outcome that mattered was the architecture contrast, and the pre-registered reading was
explicit: if LI and SV tie under a model-free policy because the policy cannot exploit finer premise
ranking, then a retrieval-native 7B should break the tie. **It did not** — +0 of 327, 17 gained and
17 lost, p = 1.0000, the most symmetric null the data could have produced. See
[Results — Tier 1](#results--tier-1-frozen-real-prover-v1-7b).

`VLLMPolicy` implements the same `TacticPolicy` protocol as the model-free policy, so the search
harness, benchmarks, manifests, verification and significance machinery were reused unchanged — which
is why the two tiers are comparable at all.

#### The calibration gate had to be redefined, and this is why

The original plan gated Tier 1 on reproducing REAL-Prover's published **FATE-M 56.7 within ~3%**.
Reading their `conf/config.py` shows that is not affordable at any budget available here:

| configuration | nodes × samples | generations per problem |
|---|---|--:|
| their shipped default (`NUM_SAMPLES=16`, `MAX_NODES=64`) | 64 × 16 | 1,024 |
| their commented-out "large" variant | 1,024 × 64 | 65,536 |
| **the paper's Pass@64** — 64 passes of large | — | **4,194,304** |

Their 56.7 sits at roughly **4,000×** the generation budget of a single shipped-config pass. A gate
demanding that number would be failed by a perfectly correct harness, and passing it would require
about 4,000 GPU-hours per arm.

**Replacement gate — three checks, all affordable:**

1. **Floor.** The 7B model with retrieval must clear the model-free result on FATE-M (22.0%) by a
   wide margin. If a fine-tuned prover cannot beat 19 hand-written tactics, the harness is wrong.
   → **32.6%, passed** (46 problems against 31).
2. **Direction.** Retrieval must help, matching the sign of their published 56.7 vs 44.7 ablation.
   The *sign* is reproducible at our budget even though the magnitude is not.
   → **+14 pooled, p = 0.0045, passed** — and significant, which their own +1.1 ProofNet ablation
   does not establish.
3. **Mechanics.** From `PolicyStats`: a low `cheat_rate`, a low `empty_rate`, and
   `mean_candidates_per_expansion` in the high single digits or better out of 16 samples. A value
   near 1–2 means the prompt or the stop sequences are wrong, and no pass rate from that run means
   anything. → **passed**, with `mean_candidate_logprob` −0.81 to −1.00 across all six runs.

Every Tier 1 table caption states the budget explicitly. Absolute numbers are **not** comparable to
the published 56.7 / 23.7 and are not presented as if they were.

**Gate 3 needed a fourth number, and finding that out cost a run.** `mean_candidates_per_expansion`
was the only health check on the list, and it is not one: under a *wrong* chat template the model
emitted token salad, which produced **more** distinct tactics (13) than the correct template did (8),
because noise does not repeat itself. Candidate diversity looked healthiest exactly when generation
was worthless. `mean_candidate_logprob` was added for this reason — the correct template measured
−0.34 against deepseek's −2.78 on the same prompts — and it is the number that would have caught the
fault in minutes rather than after a five-problem run scored 0/5.

#### Fidelity, transcribed rather than guessed

Read from their source, not inferred — three of these corrected values I had originally guessed
wrong, and each would have degraded the model in a way that looks like a retrieval result:

| | theirs | note |
|---|---|---|
| candidate score | `cumulative_logprob / max(n_tokens, 1)` | per-token **mean**; the raw sum biases search toward short tactics |
| `temperature` | **1.5** | high on purpose — best-first reranks by logprob afterwards |
| `top_p` | 0.9 | |
| `max_tokens` | 256 | |
| `NUM_QUERYS` | 10 retrieved, 6 rendered | the prompt builder truncates |
| premise order | best-ranked **first**, i.e. furthest from the goal | opposite of the predecessor project's convention |
| chat template | `deepseek`, hard-coded | **transcribed faithfully and wrong — see below** |
| banned tactics | `sorry`, `admit`, `apply?` | our guard is a superset, adding `sorryAx` and `native_decide` |

**The one place transcribing their source was the wrong call.** Their `build_local_prompt_str`
hard-codes the `deepseek` template despite a Qwen2.5-Math base. That was read, noted as looking like a
bug in their code, and followed anyway on the reasoning that the weights saw whatever they trained
with. Measured on the model's own tokenizer, it is a bug:

| template | mean logprob | matches the shipped `chat_template`? | outcome on 5 FATE-M problems |
|---|--:|---|---|
| `qwen_chatml` (the tokenizer's own) | **−0.34** | yes | **2/5 proved**, 89 expansions |
| `qwen` (generic system prompt) | −0.35 | no | — |
| `deepseek` (theirs, hard-coded) | **−2.78** | no | **0/5**, 1 progress step, 11/96 empty completions |

Under deepseek the model emitted things like `俾 Evangel Daniel dialogueCSV refriger.diag stmt` and
bare fragments ending in `]` where tactics belong. Nothing except a direct comparison against the
tokenizer's own `chat_template` would have revealed it: the run completed, finalised, and reported a
plausible number. The prompt is now re-rendered through `tokenizer.apply_chat_template` once per run
and the run **hard-fails on a mismatch** rather than proceeding — `prover/prompt.py`,
`vllm_policy.check_prompt_format`.

Two related traps in the same area, both now guarded. The turn-end token `<|im_end|>` (151645) is not
the `eos_token` `<|endoftext|>` (151643), and `skip_special_tokens` strips special tokens *before*
string matching, so stopping correctly requires `stop_token_ids` rather than stop strings. And vLLM v1
leaves `CompletionOutput.cumulative_logprob` as `None` unless `SamplingParams.logprobs` is set — which
would have scored every candidate 0.0, the *maximum* for a log probability, collapsing best-first
search to alphabetical order while the run completed and reported a plausible pass rate. The generator
now raises `SystemExit` on a `None` logprob rather than ranking on it.

**Memory, since this is where it binds:** 7B weights in bf16 are ~15 GB, and the KV cache — not the
weights — is what fills an 80 GB A100, growing with prompt length, which retrieval directly inflates.
`--max-model-len 4096` caps it. Three further engine settings are not optional on this cluster:
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and
`VLLM_USE_FLASHINFER_SAMPLER=0` — the last because FlashInfer JIT-compiles a kernel and GPU compute
nodes have no `nvcc`. See [design notes](#design-notes) for the fork-poisoning fault the first two
work around.

### H4 — the two architectures are complementary, so fusion should beat both

> **Claim.** SV and LI tie on counts while disagreeing on problems, so a ranking that combines them
> closes more theorems than either alone.

**Motivated, not tested — and Tier 1 made it the single best-supported next experiment.** Under the
frozen 7B the oracle union is 57/141 on FATE-M (+11 over either arm) and 32/186 on ProofNet (+6),
pooling to **89/327 against 72** — a ceiling of **+17 while retrieval's entire measured effect is
+14**. Track A′ agrees in sign at a smaller magnitude: 38/141 (+3) and 24/186 (+4), see
[finding 5](#5-equal-counts-different-theorems).

T3 also says *why* fusion is the right shape of fix rather than a better single retriever: the two
architectures produce differently-conditioned generations whose advantage lives in the mean rather
than the max, so widening the candidate pool at the top of the ranking — which is what fusion does —
attacks exactly the quantity best-first search consumes.

**Test.** A `fusion` arm doing reciprocal-rank fusion over both rankings, or simply interleaving both
top-k lists. One run per benchmark, no new retriever, no training. Outcomes:

* **Fusion beats both arms** → the null in finding 3 is about *counts*, not about information: the two
  architectures carry different signal and the practical recommendation becomes "use both", at SV's
  latency plus LI's.
* **Fusion matches the better arm** → the disagreement is noise in the search rather than a
  difference in retrieval quality, which would sharpen the null considerably.

### H7 — the null is two populations cancelling rather than noise ✅ ANSWERED

> **Claim.** The predecessor study measured LI winning on *novel* premises and SV winning
> in-distribution. If that ordering survives into proving, LI's 17 exclusive wins are the problems
> whose proofs need premises the retriever never trained on and SV's 17 are the problems needing
> premises it did — so +0 is a cancellation, not an absence.

**Ruled out, and the reason is more useful than the claim would have been.** The three outcomes were
written down before running it:

| outcome | reading | status |
|---|---|---|
| LI's wins enriched for unseen premises | the null is two populations; the predecessor's claim survives end to end | ❌ ruled out (+0.031, p = 0.70) |
| no enrichment, and proofs *do* need novel premises | LI's robustness advantage genuinely fails to transfer | ❌ ruled out (proofs are 2.6× in-distribution) |
| no enrichment because proofs rarely need novel premises | the regime LI wins in is barely exercised; the null is conditional, not general | ✅ **this one** |

See [T4](#t4-why-the-predecessors-robustness-advantage-does-not-transfer). The conclusion is
conditional rather than negative: late interaction is not shown to be useless, it is shown to be
**unexercised** on these benchmarks, which is a claim about the evaluation distribution and not about
the architecture.

**What would exercise it.** A benchmark whose proofs cite premises outside the retriever's training
distribution — deliberately constructed by holding out a Mathlib area from the retriever's training
set and evaluating on theorems from it, which is the `novel_premises` design applied end to end
rather than at the retrieval level. That is a data-construction task, not a compute task, and it is
the experiment this project would run next given more than six weeks.

### H8 — the architecture null is stable under re-sampling

> **Claim.** The 17/17 split is a property of the two retrievers, not of one sample. Re-running both
> arms with fresh sampling reproduces Δ ≈ 0 within noise, and the standard deviation of the LI−SV
> difference is small relative to the +14 that retrieval itself buys.

**This is the weakest point in the current evidence and it is the reason Phase 2 comes before the
more interesting phases.** Tier 1 ran *one* pass per arm at `temperature 1.5` with sampling
deliberately unseeded (`--sampling-seed` is off by default so pass@k stays honest). Every paired test
reported above treats a problem's outcome as fixed given its arm — and for a temperature-1.5 language
model it is not. The 17/17 symmetry is one draw from a distribution whose spread has never been
measured.

**Two further claims are more fragile than Δ itself, and they fail together.** A null is not
manufactured by variance — noise hides effects, it does not invent zeros — so Δ = +0 is comparatively
safe. But these are not:

* **"Equal counts, different theorems"** ([T5](#t6-equal-counts-different-theorems--and-the-ceiling-exceeds-the-effect)) rests on 17 problems being solved by exactly one arm.
* **The fusion ceiling of +17** ([H4](#h4--the-two-architectures-are-complementary-so-fusion-should-beat-both)) rests on the union reaching 89.

If re-running **one arm against itself** also flips ~17 problems and its own two draws also union to
~+17, then both claims describe the sampler rather than the retrievers, and both have to be withdrawn.
Nothing in the current data can tell: with one draw per arm there is no estimate of what a re-run does
on its own.

**Test.** Re-run `sv` and `li` on both benchmarks at two further seeds — 8 runs, ~17.7 GPU-hours —
and measure the **noise floor**: the same arm, the same configuration, a different sampling draw.
`scripts/replication_variance.py` then reports every between-arm quantity against that floor. It also
computes the estimand a single draw only approximates: each problem's *solve rate* across draws,
paired between arms.

The script refuses to report rather than mislead in two cases — two runs sharing a seed, and two
draws whose shared proofs are **byte-identical**, which would mean the seed never reached the sampler
and every variance below it is a measurement of nothing.

Outcomes:

* **Floor is small, Δ stays near 0** → the null is a property of the retrievers, now with an error
  bar instead of an assumption, and T5 and H4 survive.
* **Floor is comparable to 17** → "different theorems" and the fusion ceiling are resampling headroom.
  Two of the more interesting statements in this README come out, and H4 stops being the
  best-supported next experiment. Uncomfortable, and far better found here than by an examiner.
* **Δ swings by several problems between draws** → every single-draw paired test overstates its
  precision and each Tier 1 contrast needs restating as a mean over draws.

This is the only phase that can *invalidate* a number already published, which is why it goes first.

### H5 — the null is a property of the search width, not of the retrievers

> **Claim.** LI's advantage over SV lies in the *mean* of the candidate distribution, not its
> maximum. A best-first search that samples only 16 tactics per state reads the maximum and discards
> the rest, so widening the sample count should convert LI's advantage into proofs, and SV should gain
> less from the same widening.

This is the prediction [T3](#t3-why-the-architectures-tie-the-search-reads-the-part-retrieval-does-not-improve)
makes, stated before anyone tests it. The measured asymmetry is the whole basis: retrieval moves the
mean candidate 3–8× more than the best one, and between the two architectures the best candidate is
not improved at all (−0.0102 on FATE-M, −0.0043 on ProofNet, both non-significant).

**Test.** Re-run SV and LI on FATE-M at `--samples-per-step 32` and 64, leaving everything else fixed.
Three outcomes, all informative:

* **LI gains more than SV** → the architecture null is an artefact of search width, the retrieval
  difference is real and exploitable, and the practical guidance becomes a *joint* one about retriever
  and search budget rather than about the retriever alone. This is the outcome the mechanism predicts.
* **Both gain equally** → LI's distributional edge is not usable at any width, which makes the null a
  statement about the retrievers and closes the question much more firmly than the current evidence
  does.
* **Neither gains** → the binding constraint is the generator's ability to use premises at all, not
  candidate ranking, and effort should move to Track B rather than to retrieval.

Cost is the reason this is stated rather than run: a FATE-M LI arm at 32 samples is roughly twice the
2.1 h the 16-sample run took, and the 6-week budget is spent.

### H6 — a retriever-agnostic generator removes the last confound

> **Claim.** REAL-Prover-v1 was trained on `LeanSearch-PS` output, so both ProofLens retrievers are
> out-of-distribution for it. A generator trained with each retriever's own premises would show LI's
> true end-to-end value.

This is Track B in the original plan — LoRA SFT on Qwen3-4B and Kimina-Prover-Distill-1.7B, with
byte-identical hyperparameters and step counts per arm, retrieval attached once per arm and cached. It
is the one remaining design that removes the handicap noted in the limitations, and it is the only way
to distinguish "late interaction does not help this prover" from "late interaction does not help a
prover that was trained to expect a different retriever". Not started; ~2 weeks of GPU time.

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

### 3. Run an arm — Track A′ (model-free)

```bash
python scripts/prove_benchmark.py --benchmark fate_m --arm li \
    --index data/index/li_ft_novel_bm25 --data-root <REAL-Prover>/data \
    --lean-project ~/lean/mathlib_v4160 --samples-per-step 32 --min-closers 19 \
    --n-candidates 50000
```

Arms: `none` · `bm25` · `sv` · `li`. Everything except `--arm` is held fixed and recorded in the run
manifest, so two runs are comparable exactly when their manifests differ only in the arm.

**`--n-candidates 50000` is required to reproduce the LI number above** and applies to the `li` arm
only. It overrides the first-stage budget stored in the index (1,000) without rebuilding the 5.5 GB
index; the value used is recorded in the manifest, and `compare_arms.py` labels the arm with it
(`li@50k`). Omitting it reproduces the 22/141 run instead — the same code, a different experiment.

`--min-closers 19` reserves candidate slots for the shared repertoire before premise tactics
compete. Without it, premise tactics displace closers and a *better* retriever can score *worse* —
observed as `none` 10/30 against `bm25` 6/30, where bm25's solved set was a strict subset of none's.

On SLURM:

```bash
BENCHMARK=fate_m ARM=li N_CANDIDATES=50000 \
    sbatch -p <gpu-partition> -A <account> -G 1 slurm/prove_benchmark.sbatch
```

Confirm the budget from the run's **manifest**, not the job header: the header echoes a shell
variable, the manifest records what the retriever actually used. A five-hour run was lost to the
difference (see `LEARNINGS.md`).

### 3b. Run an arm — Tier 1 (frozen 7B)

Needs a GPU with ≥40 GB, and a **separate environment** from the retrieval one — vLLM pins `torch`
hard, which is why the `all` extra deliberately excludes it:

```bash
bash scripts/bootstrap_llm.sh          # idempotent; ends in ENV OK for both environments
```

It installs `.[serve]`, then adds `lean-interact` and `pylate` under a **constraints file** pinning
`torch` to the version vLLM was compiled against, so pip resolves real dependency trees without
silently downgrading torch. `scripts/check_env.py` then imports every third-party module the codebase
imports at any nesting depth — found by walking the source with `ast`, because most are imported
inside functions to keep the tests hermetic, so `import prooflens_prover` proves nothing.

**Preflight first — it is a hard gate and takes 30 minutes.** It loads the model and samples once with
no Lean and no retrieval, so a prompt or engine fault surfaces before a multi-hour run consumes a GPU:

```bash
python scripts/preflight_llm.py --model FrenzyMath/REAL-Prover \
    --compare-templates qwen_chatml,qwen,deepseek
```

It prints one row per template — whether it matches the tokenizer's own `chat_template`, how many
distinct tactics it produced, how many were garbled, and the mean candidate log-probability — and
**fails outright on a chat-template mismatch**. `qwen_chatml` must win; if it does not, stop.

```bash
python scripts/prove_benchmark.py --benchmark fate_m --arm li --policy vllm \
    --model FrenzyMath/REAL-Prover --template qwen_chatml \
    --index data/index/li_ft_novel_bm25 --data-root <REAL-Prover>/data \
    --lean-project ~/lean/mathlib_v4160 --n-candidates 50000 \
    --samples-per-step 16 --top-k-premises 10 --max-model-len 4096
```

On SLURM:

```bash
BENCHMARK=fate_m ARM=li N_CANDIDATES=50000 \
    sbatch -p <gpu-partition> -A <account> -G 1 slurm/prove_benchmark_llm.sbatch
```

Sampling defaults are REAL-Prover's own — `temperature 1.5`, `top_p 0.9`, `max_tokens 256`,
`samples_per_step 16`, `max_nodes 64` — and are derived from `SamplingConfig` rather than restated in
argparse, because an argparse default is *always* passed and silently overrode the corrected values
once already.

**Check `mean_candidate_logprob` in the run summary before believing a pass rate.** Healthy is roughly
−0.3 to −1.0; below about −1.5 the prompt or the stop tokens are wrong and the pass rate is
meaningless. Candidate *count* is not a health check — a wrong template produced more distinct tactics
than the right one.

### 4. Verify, then analyse

```bash
python scripts/verify_proofs.py --run results/logs/<run_id> \
    --data-root <REAL-Prover>/data --lean-project ~/lean/mathlib_v4160

python scripts/build_table1.py --policy vllm          # Tier 1 table
python scripts/build_table1.py --policy repertoire    # Track A' table

python scripts/compare_arms.py --baseline results/logs/<sv_run> \
                               --treatment results/logs/<li_run>
```

`build_table1.py` discovers runs rather than taking ids, so a control cannot be paired against the
wrong benchmark's treatment. `--policy` is required to be explicit because mixing a model-free run
and an LLM run into one table would be meaningless.

**Pooling two benchmarks into one paired test**, which is what makes the Tier 1 retrieval result
significant:

```bash
python scripts/compare_arms.py \
    --baseline results/logs/<fate_none> --treatment results/logs/<fate_li> \
    --baseline results/logs/<pn_none>   --treatment results/logs/<pn_li>
```

Repeat the flags in matching order; every per-benchmark row is printed alongside the pooled figure,
and a sign disagreement between them is flagged loudly, because a pooled average over benchmarks
chosen for *different* retrieval sensitivity can describe neither of them.

**Root-state candidate quality** — the mechanism behind the architecture null. No GPU, no model, no
re-run:

```bash
python scripts/root_candidate_quality.py \
    --run results/logs/<none_run> --run results/logs/<sv_run> --run results/logs/<li_run>
```

This one needs `results/logs`, **not** `results/exported/logs`: it reads the per-tactic `trace`, which
is the bulk of `attempts.jsonl` and is stripped from the exported records. Its output
(`results/exported/tables/root_quality_*.json`) ships so the numbers are auditable, but reproducing them from
scratch means re-running the arms.

**Replication variance** — Phase 2. Re-run an arm at a different `SEED` and the noise floor becomes
measurable:

```bash
for SEED in 1 2; do
  for ARM in sv li; do
    BENCHMARK=fate_m ARM=$ARM SEED=$SEED N_CANDIDATES=50000 \
        sbatch -p <gpu-partition> -A <account> -G 1 slurm/prove_benchmark_llm.sbatch
  done
done

python scripts/replication_variance.py \
    --run results/logs/<sv_seed0> --run results/logs/<sv_seed1> --run results/logs/<sv_seed2> \
    --run results/logs/<li_seed0> --run results/logs/<li_seed1> --run results/logs/<li_seed2>
```

`SEED` is the sampling draw and the **only** thing a replicate may vary; the script refuses runs that
differ in anything else, and refuses two runs sharing a seed. Pass the control arm's runs first —
arm order fixes the sign of every delta, and sorting alphabetically would silently make `li` the
baseline.

**Unseen-premise stratification** — Phase 1, the test of whether the architecture null is two
populations cancelling. Runs from the exported records, since it needs only each proof's text:

```bash
python scripts/novel_premise_stratification.py \
    --seen-split <prooflens>/leandojo_data/leandojo_benchmark_4/novel_premises/train.json \
    --corpus data/premises/mathlib_v4160.jsonl \
    --run results/exported/logs/<fate_none> --run results/exported/logs/<fate_sv> \
    --run results/exported/logs/<fate_li>   --run results/exported/logs/<pn_none> \
    --run results/exported/logs/<pn_sv>     --run results/exported/logs/<pn_li>
```

Passing runs from both benchmarks pools them, namespacing problem ids by benchmark so the two cannot
collide — the LI-vs-SV contrast has only 11 and 6 exclusive wins per arm separately, and pooling to 17
is the difference between a test and a gesture. `--seen-split` is parsed once (365 MB, ~30 s) and
cached; `--drop-tactic-words` gives the sensitivity. It needs the **predecessor repository's** LeanDojo
split, which is the one input to any analysis here that this repo does not carry.

To reproduce every published table from the exported records in this repo:

```bash
python scripts/build_table1.py --results-root results/exported/logs --policy vllm
python scripts/build_table1.py --results-root results/exported/logs --policy repertoire
```

### Tests

```bash
pytest tests/ -q -m "not lean"                                    # 684, hermetic
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

**LLM policy.** `VLLMPolicy` is a `TacticPolicy` like any other arm, so the search cannot tell it from
the repertoire. Candidates are ranked on `cumulative_logprob / max(n_tokens, 1)` — a per-token mean, so
tactics of different lengths compare fairly and the search is not biased toward short ones. Prompts are
left-truncated by token id when they exceed `max_model_len - max_tokens`, keeping the *most recent*
context, because this cluster's vLLM rejects `truncate_prompt_tokens` outright. Tactics containing
U+FFFD are dropped before the cheat guard: Lean cannot parse a replacement character, and a mid-token
truncation is how one appears.

**Fork poisoning.** `torch.cuda.is_available()` registers a `pthread_atfork` handler that marks every
subsequently forked child as unable to use CUDA, while initialising nothing — so
`torch.cuda.is_initialized()` stays `False`. That is the exact blind spot in vLLM's
`_maybe_force_spawn`, and a
seeding helper calling `is_available()` was enough to make the engine die with `Cannot re-initialize
CUDA in forked subprocess`. Fixed at the root by seeding unconditionally (`torch.manual_seed` already
covers CUDA devices), with `VLLM_ENABLE_V1_MULTIPROCESSING=0` and
`VLLM_WORKER_MULTIPROC_METHOD=spawn` as independent backstops. `tests/test_seed.py` keeps a tripwire on
`cuda.is_available` so the call cannot come back.

**Reproducibility.** Every run writes a manifest — config, seed, git SHA (with a `-dirty` marker),
package versions, Lean version, hardware, SLURM job id — before doing any work, so a job killed by
the scheduler still leaves a record of what it attempted.

## Layout

```
src/prooflens_prover/
  lean/         backend protocol, LeanInteract backend, proof verdicts
  retrieval/    base, bm25, dense (SV + LI), lean tokenizer
  prover/       best-first search, model-free repertoire, vLLM policy, prompt templates
  data/         benchmark and premise-corpus loaders
  eval/         paired comparison, significance, draws (one run as one sample)
  utils/        seeding, logging, run manifests, io
scripts/        extraction, index building, benchmarks, verification, analysis
slurm/          cluster jobs
tests/          684 hermetic + 9 live-Lean
results/        exported run records and tables
```

## Acknowledgements

Benchmarks and reference numbers from [REAL-Prover](https://github.com/frenzymath/REAL-Prover)
(arXiv:2505.20613) and [FATE](https://github.com/frenzymath/FATE). Lean interaction via
[LeanInteract](https://github.com/augustepoiroux/LeanInteract). Retriever checkpoints from the
predecessor ProofLens premise-selection study.
