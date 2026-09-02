#!/usr/bin/env python3
"""Re-derive every number in README.md from the records it claims, and fail on any mismatch.

    python scripts/check_readme.py

The README is the only prose in this repository, so it is the only place a wrong number can hide.
It restates counts, p-values and intervals computed elsewhere, and a restated number drifts
silently. An earlier revision quoted the four per-benchmark p-values as 0.0923 / 0.2100 / 0.3877 /
0.3877 without naming a test: those are `results/tables/table1.md`'s **exact McNemar** column, while
the README's own significance rule is the bootstrap-plus-permutation pair, which gives 0.094 / 0.202
/ 0.391 / 0.390. Both were correct numbers; the pairing was not, and nothing noticed, because a
p-value carries no label saying which test produced it. This script pins the permutation values and
the README now names the test.

So everything the README puts in a table is recomputed here from `results/tables/*.json` and the
exported run records -- the same code paths `passk_union.py` and `budget_matched.py` use -- and
compared. Also checked: that every figure the README embeds exists, and that no working-notes file
is referenced by name.

Hermetic apart from the exported records: no GPU, no Lean, no model, no network.
Exit status is non-zero on any mismatch, so this can gate a publish.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import figures_data as D  # noqa: E402
from budget_matched import passk  # noqa: E402
from passk_profile import arm_of  # noqa: E402
from passk_union import discover  # noqa: E402
from prooflens_prover.eval.draws import load_draw  # noqa: E402

ROOT = REPO / "results" / "exported" / "logs"
TXT = (REPO / "README.md").read_text(encoding="utf-8")
ok = fail = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok, fail
    if condition:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


def says(label: str, needle: str) -> None:
    check(label, needle in TXT, f"missing {needle!r}")


def table(name: str) -> dict:
    return json.loads((REPO / "results" / "tables" / name).read_text(encoding="utf-8"))


# --- Tier 1: the headline contrast --------------------------------------------------------------
t1 = table("table1.json")
c = t1["comparisons"]
for key, side, want in (("fate_m:sv_vs_none", "baseline", 39),
                        ("fate_m:sv_vs_none", "treatment", 46),
                        ("fate_m:li_vs_none", "treatment", 46),
                        ("proofnet_test:sv_vs_none", "baseline", 24),
                        ("proofnet_test:sv_vs_none", "treatment", 28),
                        ("proofnet_test:li_vs_none", "treatment", 28)):
    check(f"tier1 {key}/{side} = {want}", c[key][side]["proved"] == want,
          f"actual {c[key][side]['proved']}")
check("tier1 pooled none = 63",
      c["fate_m:sv_vs_none"]["baseline"]["proved"]
      + c["proofnet_test:sv_vs_none"]["baseline"]["proved"] == 63)
check("tier1 union fate = 56", t1["oracle_union"]["fate_m"]["n_union"] == 56)
check("tier1 union pnet = 32", t1["oracle_union"]["proofnet_test"]["n_union"] == 32)

for key, p in (("fate_m:sv_vs_none", 0.094), ("fate_m:li_vs_none", 0.202),
               ("proofnet_test:sv_vs_none", 0.391), ("proofnet_test:li_vs_none", 0.390)):
    got = c[key]["p_permutation"]
    check(f"tier1 p({key}) = {p}", abs(got - p) < 0.001, f"actual {got:.4f}")

pooled = {n: table(f"{f}.json")["pooled"]["primary"] for n, f in
          (("sv", "pooled_sv_vs_none"), ("li", "pooled_li50k_vs_none"),
           ("lisv", "pooled_li50k_vs_sv"))}
check("pooled sv = +11", pooled["sv"]["delta_problems"] == 11)
check("pooled li = +11", pooled["li"]["delta_problems"] == 11)
check("pooled li-sv = +0", pooled["lisv"]["delta_problems"] == 0)
check("pooled sv p = 0.038", abs(pooled["sv"]["p_permutation"] - 0.0381) < 0.0006)
check("pooled li p = 0.089", abs(pooled["li"]["p_permutation"] - 0.0890) < 0.0006)
# Displacement: the split, not the net, is what costs late interaction its significance.
for name, gained, lost in (("sv", 18, 7), ("li", 23, 12), ("lisv", 14, 14)):
    d = pooled[name]
    check(f"pooled {name} gained = {gained}", len(d["only_treatment"]) == gained,
          f"actual {len(d['only_treatment'])}")
    check(f"pooled {name} lost = {lost}", len(d["only_baseline"]) == lost,
          f"actual {len(d['only_baseline'])}")

# --- the two arms are not one run counted twice ---------------------------------------------------
# Recomputed on the *staged* runs named in table1.json. The committed arm_distinctness_*.json
# predate the staging fix and must not be used (they say 22.9% / 45.0%).
for bench, ident_pct, discordant in (("fate_m", 16.7, 20), ("proofnet_test", 37.5, 8)):
    proofs = {}
    for arm in ("sv", "li"):
        proofs[arm] = {
            r["problem_id"]: json.dumps(r.get("proof"))
            for r in D.read_jsonl(ROOT / t1["runs"][f"{bench}/{arm}"] / "attempts.jsonl")
            if r.get("proved")
        }
    both = set(proofs["sv"]) & set(proofs["li"])
    same = sum(1 for p in both if proofs["sv"][p] == proofs["li"][p])
    check(f"{bench}: {ident_pct}% of proofs identical",
          abs(100 * same / len(both) - ident_pct) < 0.1, f"actual {100 * same / len(both):.1f}%")
    check(f"{bench}: {discordant} discordant",
          len(set(proofs["sv"]) ^ set(proofs["li"])) == discordant)

# --- Track A': the model-free replication ---------------------------------------------------------
ta = table("track_a_prime/table1.json")["comparisons"]
for key, side, want in (("fate_m:sv_vs_none", "baseline", 12),
                        ("fate_m:sv_vs_none", "treatment", 35),
                        ("fate_m:li_vs_none", "treatment", 31),
                        ("proofnet_test:sv_vs_none", "baseline", 9),
                        ("proofnet_test:sv_vs_none", "treatment", 20),
                        ("proofnet_test:li_vs_none", "treatment", 20),
                        ("minif2f_test:sv_vs_none", "baseline", 78),
                        ("minif2f_test:sv_vs_none", "treatment", 77),
                        ("minif2f_test:li_vs_none", "treatment", 79)):
    check(f"track A' {key}/{side} = {want}", ta[key][side]["proved"] == want,
          f"actual {ta[key][side]['proved']}")
for key, p in (("fate_m:sv_vs_none", 0.0001), ("fate_m:li_vs_none", 0.0002),
               ("proofnet_test:sv_vs_none", 0.0011)):
    check(f"track A' p({key}) = {p}", abs(ta[key]["p_permutation"] - p) < 0.0002,
          f"actual {ta[key]['p_permutation']:.4f}")

# --- the pass@8 sweep, from the exported records --------------------------------------------------
sweep = {}
for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
    for d in discover(ROOT, bench, "vllm",
                      ["search.samples_per_step=32",
                       "policy_config.premise_free_fraction=0.25", f"n_problems={n}"]):
        draw = load_draw(d)
        sweep[(bench, arm_of(draw), draw.seed)] = set(draw.solved)

for bench, arm, want in (("proofnet_test", "li", 39), ("proofnet_test", "sv", 39),
                         ("fate_m", "li", 68), ("fate_m", "sv", 66)):
    got = len(set().union(*(sweep[(bench, arm, s)] for s in range(8))))
    check(f"{bench} {arm}@8 = {want}", got == want, f"actual {got}")
for bench, want in (("proofnet_test", 44), ("fate_m", 72)):
    ens = set().union(*(sweep[(bench, "li", s)] | sweep[(bench, "sv", s)] for s in range(8)))
    check(f"{bench} ensemble@8 = {want}", len(ens) == want, f"actual {len(ens)}")
check("pooled ensemble@8 = 116",
      sum(len(set().union(*(sweep[(b, "li", s)] | sweep[(b, "sv", s)] for s in range(8))))
          for b in ("proofnet_test", "fate_m")) == 116)
for arm, want in (("li", 64), ("sv", 62), ("fusion", 64)):
    got = len(set().union(*(sweep[("fate_m", arm, s)] for s in range(4))))
    check(f"fate {arm}@4 = {want}", got == want, f"actual {got}")

# --- the equal-budget control ---------------------------------------------------------------------
# One ensemble draw is both arms at one seed, so it costs two generations' worth. ensemble@4 is
# therefore the budget-matched rival of single@8, not of single@4.
total = {}
for name in ("li", "sv", "ens"):
    s = 0.0
    for bench in ("proofnet_test", "fate_m"):
        li = [sweep[(bench, "li", i)] for i in range(8)]
        sv = [sweep[(bench, "sv", i)] for i in range(8)]
        draws, k = {"li": (li, 8), "sv": (sv, 8),
                    "ens": ([li[i] | sv[i] for i in range(8)], 4)}[name]
        s += sum(passk(draws, p, k) for p in D.problem_ids(ROOT, bench))
    total[name] = s
check("li@8 = 107.00", abs(total["li"] - 107.0) < 0.01, f"{total['li']:.2f}")
check("sv@8 = 105.00", abs(total["sv"] - 105.0) < 0.01, f"{total['sv']:.2f}")
check("ensemble@4 = 109.87", abs(total["ens"] - 109.87) < 0.01, f"{total['ens']:.2f}")
check("ensemble - li = +2.87", abs((total["ens"] - total["li"]) - 2.87) < 0.01,
      f"{total['ens'] - total['li']:.2f}")

# --- the two-stage approximation ------------------------------------------------------------------
rec = D.table("li_recall_fate_m")["recall_by_n_candidates"]
for n, r, lossless in (("1000", 0.443, 9), ("20000", 0.888, 81), ("50000", 0.979, 124)):
    check(f"recall@10 at n={n} is {r}", abs(rec[n]["recall"] - r) < 0.0006,
          f"actual {rec[n]['recall']:.4f}")
    check(f"lossless queries at n={n} is {lossless}", rec[n]["n_lossless"] == lossless)

# --- root candidate quality: the mechanism --------------------------------------------------------
for bench, key in (("fate_m", "fate_m"), ("proofnet", "proofnet")):
    rq = D.table(f"root_quality_{bench}")
    check(f"root quality table loads for {key}", bool(rq))

# --- contamination, bounded -----------------------------------------------------------------------
con = table("contamination_tier1.json")
won = one_step = 0
for bench in ("fate_m", "proofnet_test"):
    for arm in ("sv", "li@50k"):
        won += con[bench][arm]["won_vs_control"]
        one_step += len(con[bench][arm]["won_vs_control_by_one_step_citation"])
check("retrieval won 41 problems", won == 41, f"actual {won}")
check("7 of them were one-step corpus answers", one_step == 7, f"actual {one_step}")

# --- verification ---------------------------------------------------------------------------------
claims = rejected = 0
for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
    for d in discover(ROOT, bench, "vllm",
                      ["search.samples_per_step=32",
                       "policy_config.premise_free_fraction=0.25", f"n_problems={n}"]):
        if arm_of(load_draw(d)) not in ("li", "sv"):
            continue
        v = json.loads((d / "verification.json").read_text(encoding="utf-8"))
        claims += v["n_claimed"]
        rejected += v["n_failed"]
check("sweep claims = 1376", claims == 1376, f"actual {claims}")
check("sweep rejected = 1", rejected == 1, f"actual {rejected}")

# --- what is shipped ------------------------------------------------------------------------------
runs = [d for d in ROOT.iterdir() if (d / "manifest.json").exists()]
check("79 exported runs", len(runs) == 79, f"actual {len(runs)}")
figures = sorted((REPO / "figures").glob("*.png"))
check("17 figures", len(figures) == 17, f"actual {len(figures)}")
import re  # noqa: E402

for ref in re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", TXT):
    check(f"embedded {ref} exists", (REPO / ref).exists())

# --- strings the README must contain --------------------------------------------------------------
for needle in ("44 / 186 = 23.7%", "72 / 141 = 51.1%", "**32,768**", "4,194,304",
               "+2.87", "[−2.34, +8.54]", "0.3251", "109.87", "105.00", "107.00",
               "276,070", "276070:31db61c63a9b7ee1", "21,752,080", "0.443", "0.979",
               "1,071 hermetic", "17 figures"):
    says(f"README states {needle!r}", needle)

# --- and must not ---------------------------------------------------------------------------------
# Working notes are not published (see publish.sh's allowlist), so naming one leaves a dead pointer.
# The stale numbers are the pre-staging run set and the pre-discount equal-budget figure.
for banned, why in (("dissertation.md", "unpublished working file"),
                    ("CLAUDE", "unpublished working file"),
                    ("MEETING", "unpublished working file"),
                    ("LEARNINGS", "unpublished working file"),
                    ("DECISIONS.md", "unpublished working file"),
                    ("prooflens_results.md", "predecessor's file, not in this repo"),
                    (" 19/186", "pre-staging ProofNet control"),
                    ("110.16", "pre-discount equal-budget figure")):
    check(f"README does not mention {banned!r} ({why})", banned not in TXT)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
