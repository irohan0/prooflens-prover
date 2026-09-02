#!/usr/bin/env python3
"""Cross-check every headline number in dissertation.md against the records it claims.

    python scripts/check_dissertation.py

The file `dissertation.md` is the master writing source: a long document that restates numbers
computed elsewhere. Restated numbers go stale silently -- three did, and were caught only by
re-deriving them: a pre-staging control count, an undecodable-count trend the staged runs do not
support, and an equal-budget figure computed before the verification discount was applied.

So every count, p-value and interval the document puts in bold is re-derived here from
`results/tables/*.json` and the exported run records, and compared. Also checked: that all seventeen
figures exist, are each referenced exactly once, appear in numerical order, and that no file in
`figures/` is orphaned.

Exit status is non-zero on any mismatch, so this can gate a commit.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import figures_data as D  # noqa: E402
from budget_matched import passk  # noqa: E402
from passk_profile import arm_of  # noqa: E402
from passk_union import discover  # noqa: E402
from prooflens_prover.eval.draws import load_draw  # noqa: E402

ROOT = pathlib.Path("results/exported/logs")
TXT = pathlib.Path("dissertation.md").read_text(encoding="utf-8")
ok = fail = 0


def check(label, condition, detail=""):
    global ok, fail
    if condition:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


def says(label, needle):
    check(label, needle in TXT, f"missing: {needle!r}")


# --- Tier 1, from the pinned table -----------------------------------------------------------
t1 = json.loads(pathlib.Path("results/tables/table1.json").read_text(encoding="utf-8"))
c = t1["comparisons"]
check("fate none=39", c["fate_m:sv_vs_none"]["baseline"]["proved"] == 39)
check("fate sv=46", c["fate_m:sv_vs_none"]["treatment"]["proved"] == 46)
check("fate li=46", c["fate_m:li_vs_none"]["treatment"]["proved"] == 46)
check("pnet none=24", c["proofnet_test:sv_vs_none"]["baseline"]["proved"] == 24)
check("pnet sv=28", c["proofnet_test:sv_vs_none"]["treatment"]["proved"] == 28)
check("pnet li=28", c["proofnet_test:li_vs_none"]["treatment"]["proved"] == 28)
check("fate union=56", t1["oracle_union"]["fate_m"]["n_union"] == 56)
check("pnet union=32", t1["oracle_union"]["proofnet_test"]["n_union"] == 32)
for key, p in (("fate_m:sv_vs_none", 0.094), ("fate_m:li_vs_none", 0.202),
               ("proofnet_test:sv_vs_none", 0.391), ("proofnet_test:li_vs_none", 0.390)):
    check(f"{key} p={p}", abs(c[key]["p_permutation"] - p) < 0.0006,
          f"actual {c[key]['p_permutation']:.4f}")

pooled = {n: json.loads(pathlib.Path(f"results/tables/{f}.json").read_text(encoding="utf-8"))
          ["pooled"]["primary"] for n, f in
          (("sv", "pooled_sv_vs_none"), ("li", "pooled_li50k_vs_none"),
           ("lisv", "pooled_li50k_vs_sv"))}
check("pooled sv d=+11", pooled["sv"]["delta_problems"] == 11)
check("pooled sv p=0.038", abs(pooled["sv"]["p_permutation"] - 0.0381) < 0.0006)
check("pooled li p=0.089", abs(pooled["li"]["p_permutation"] - 0.0890) < 0.0006)
check("pooled li-sv d=0", pooled["lisv"]["delta_problems"] == 0)
check("pooled li-sv CI +/-3.06",
      abs(100 * pooled["lisv"]["ci95"][1] - 3.06) < 0.02,
      f"actual {100 * pooled['lisv']['ci95'][1]:.2f}")

# --- discordance -----------------------------------------------------------------------------
for bench, f, sv_only, li_only in (("proofnet_test", "discordance_proofnet", 4, 4),
                                   ("fate_m", "discordance_fate_m", 10, 10)):
    d = json.loads(pathlib.Path(f"results/tables/{f}.json").read_text(encoding="utf-8"))
    check(f"{bench} only_sv={sv_only}", d["only_sv"]["n"] == sv_only)
    check(f"{bench} only_li={li_only}", d["only_li@50k"]["n"] == li_only)

# --- sweep, from the exported records ---------------------------------------------------------
sweep = {}
for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
    dirs = discover(ROOT, bench, "vllm",
                    ["search.samples_per_step=32",
                     "policy_config.premise_free_fraction=0.25", f"n_problems={n}"])
    for p in dirs:
        dr = load_draw(p)
        sweep[(bench, arm_of(dr), dr.seed)] = set(dr.solved)

for bench, arm, want in (("proofnet_test", "li", 39), ("proofnet_test", "sv", 39),
                         ("fate_m", "li", 68), ("fate_m", "sv", 66)):
    got = len(set().union(*(sweep[(bench, arm, s)] for s in range(8))))
    check(f"{bench} {arm}@8 = {want}", got == want, f"actual {got}")

for bench, want in (("proofnet_test", 44), ("fate_m", 72)):
    ens = set().union(*(sweep[(bench, "li", s)] | sweep[(bench, "sv", s)] for s in range(8)))
    check(f"{bench} ensemble@8 = {want}", len(ens) == want, f"actual {len(ens)}")

for arm, want in (("li", 64), ("sv", 62), ("fusion", 64)):
    got = len(set().union(*(sweep[("fate_m", arm, s)] for s in range(4))))
    check(f"fate {arm}@4 = {want}", got == want, f"actual {got}")
ens4 = set().union(*(sweep[("fate_m", "li", s)] | sweep[("fate_m", "sv", s)] for s in range(4)))
check("fate ensemble@4 = 68", len(ens4) == 68, f"actual {len(ens4)}")

# --- equal-budget expectations ------------------------------------------------------------------
tot = {}
for name in ("li", "sv", "ens"):
    s = 0.0
    for bench in ("proofnet_test", "fate_m"):
        ids = D.problem_ids(ROOT, bench)
        li = [sweep[(bench, "li", i)] for i in range(8)]
        sv = [sweep[(bench, "sv", i)] for i in range(8)]
        pool, k = {"li": (li, 8), "sv": (sv, 8),
                   "ens": ([li[i] | sv[i] for i in range(8)], 4)}[name]
        s += sum(passk(pool, p, k) for p in ids)
    tot[name] = s
check("equal-budget li@8 = 107.00", abs(tot["li"] - 107.0) < 0.01, f"{tot['li']:.2f}")
check("equal-budget sv@8 = 105.00", abs(tot["sv"] - 105.0) < 0.01, f"{tot['sv']:.2f}")
check("equal-budget ens@4 = 109.87", abs(tot["ens"] - 109.87) < 0.01, f"{tot['ens']:.2f}")

# --- verification totals ------------------------------------------------------------------------
claims = rejected = 0
for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
    for p in discover(ROOT, bench, "vllm",
                      ["search.samples_per_step=32",
                       "policy_config.premise_free_fraction=0.25", f"n_problems={n}"]):
        if arm_of(load_draw(p)) not in ("li", "sv"):
            continue
        v = json.loads((p / "verification.json").read_text(encoding="utf-8"))
        claims += v["n_claimed"]
        rejected += v["n_failed"]
check("sweep claims = 1376", claims == 1376, f"actual {claims}")
check("sweep rejected = 1", rejected == 1, f"actual {rejected}")

# --- cheats -------------------------------------------------------------------------------------
tc = tg = 0
for d in ROOT.iterdir():
    mf = d / "manifest.json"
    if not mf.exists():
        continue
    m = json.loads(mf.read_text(encoding="utf-8"))
    if m["config"].get("policy_kind") != "vllm":
        continue
    ps = (m.get("outcome") or {}).get("policy_stats") or {}
    tc += ps.get("n_cheats") or 0
    tg += ps.get("n_generated") or 0
check("total cheats = 47", tc == 47, f"actual {tc}")
check("total generations = 8,347,424", tg == 8347424, f"actual {tg:,}")


# --- the METHOD chapter against the source, not against memory ------------------------------------
# Every constant section 4 states is imported and compared. A default changed in code without the
# document following would otherwise be invisible: the numbers in section 5-9 would still verify,
# because they come from manifests, while the method describing how they were produced drifted.
from prooflens_prover.data.premises import DEFAULT_KINDS  # noqa: E402
from prooflens_prover.lean.backend import TacticPolicy  # noqa: E402
from prooflens_prover.prover.prompt import DEFAULT_TEMPLATE, TACTIC_INSTRUCTION  # noqa: E402
from prooflens_prover.prover.repertoire import DEFAULT_CLOSERS, DEFAULT_TEMPLATES  # noqa: E402
from prooflens_prover.prover.search import SearchConfig  # noqa: E402
from prooflens_prover.prover.vllm_policy import SamplingConfig, VLLMPolicy  # noqa: E402
from prooflens_prover.retrieval.base import DEFAULT_TOP_K, PROMPT_PREMISE_LIMIT  # noqa: E402
from prooflens_prover.retrieval.dense import (  # noqa: E402
    LI_DOCUMENT_LENGTH,
    LI_QUERY_LENGTH,
    SV_MAX_SEQ_LENGTH,
)
from prooflens_prover.retrieval.fusion import DEFAULT_FETCH_K, MODES, RRF_K  # noqa: E402

_s, _sc, _g = SearchConfig(), SamplingConfig(), TacticPolicy()
_pol = VLLMPolicy.__dataclass_fields__
for label, got, want in [
    ("search 64x16", (_s.max_expansions, _s.samples_per_step), (64, 16)),
    ("max_depth 32", _s.max_depth, 32),
    ("wall clock 600", _s.wall_clock_s, 600.0),
    ("tactic timeout 60", _s.tactic_timeout, 60.0),
    ("alpha 0.5", _s.length_penalty, 0.5),
    ("dedupe states", _s.dedupe_states, True),
    ("large 1024x64", (SearchConfig.large().max_expansions,
                       SearchConfig.large().samples_per_step), (1024, 64)),
    ("temperature 1.5", _sc.temperature, 1.5),
    ("top_p 0.9", _sc.top_p, 0.9),
    ("max_tokens 256", _sc.max_tokens, 256),
    ("logprobs 1", _sc.logprobs, 1),
    ("stop excludes newline", chr(10) in _sc.stop, False),
    ("turn end token", list(_sc.turn_end_tokens), ["<|im_end|>"]),
    ("native_decide banned", _g.allow_native_decide, False),
    ("max tactic chars 2000", _g.max_tactic_chars, 2000),
    ("apply? banned", _g.reject_reason("apply? x") is not None, True),
    ("sorry banned", _g.reject_reason("sorry") is not None, True),
    ("identifier containing sorry allowed",
     _g.reject_reason("exact Nat.sorryFree") is None, True),
    ("top_k 10", DEFAULT_TOP_K, 10),
    ("premises shown 6", PROMPT_PREMISE_LIMIT, 6),
    ("premise_free default 0.0", _pol["premise_free_fraction"].default, 0.0),
    ("RRF K 60", RRF_K, 60),
    ("fusion fetch_k 32", DEFAULT_FETCH_K, 32),
    ("fusion modes", list(MODES), ["rrf", "interleave"]),
    ("LI query length 384", LI_QUERY_LENGTH, 384),
    ("LI document length 300", LI_DOCUMENT_LENGTH, 300),
    ("SV max seq 512", SV_MAX_SEQ_LENGTH, 512),
    ("template qwen_chatml", DEFAULT_TEMPLATE, "qwen_chatml"),
    ("tactic instruction", TACTIC_INSTRUCTION,
     "Please generate a tactic in lean4 to solve the state."),
    ("19 closers", len(DEFAULT_CLOSERS), 19),
    ("5 premise templates", len(DEFAULT_TEMPLATES), 5),
    ("corpus kinds", sorted(DEFAULT_KINDS),
     ["axiom", "ctor", "def", "inductive", "opaque", "theorem"]),
]:
    check(f"method: {label}", got == want, f"source={got!r} expected={want!r}")

# The `none` arm suppresses retrieval with top_k=0 rather than by a cheap call: the document says
# so, and an earlier version said the opposite.
_driver = pathlib.Path("scripts/prove_benchmark.py").read_text(encoding="utf-8")
check("method: none arm sets top_k=0", 'top_k=0 if args.arm == "none"' in _driver)
check("method: policy short-circuits on top_k",
      "if self.top_k > 0 else []" in
      pathlib.Path("src/prooflens_prover/prover/vllm_policy.py").read_text(encoding="utf-8"))
says("prose: none arm top_k = 0", "`top_k = 0` for this arm")

# --- claims the prose makes that must appear ----------------------------------------------------
for needle in ("44 / 186 (23.7%)", "72 / 141 (51.1%)", "**32,768**", "4,194,304",
               "+2.87", "[−2.34, +8.54]", "0.3251", "**107.00**", "**105.00**", "**109.87**",
               "276,070", "276070:31db61c63a9b7ee1", "187,540", "8,347,424",
               "**1,376**", "0.443", "0.979", "**1,060**"):
    says(f"prose contains {needle!r}", needle)

# --- the wall clock, which section 10.8 breaks out by arm ---------------------------------------
# Counted from `limit_hit` on every terminal attempt, because "we ran out of time" and "we ran out
# of ideas" are different findings and the document now claims the split is asymmetric at 32
# samples. Tier 1 is counted on the six canonical runs named in table1.json rather than by filter:
# the export also holds the pre-staging replicates and the li@1k run, and unioning those would
# inflate the denominators.
def _limit_hits(run_dir: pathlib.Path) -> tuple[int, int]:
    wall = total = 0
    for r in D.read_jsonl(run_dir / "attempts.jsonl"):
        total += 1
        wall += r.get("limit_hit") == "wall_clock_s"
    return wall, total


t1_wall = t1_n = 0
for _key, _rid in t1["runs"].items():
    w, n = _limit_hits(ROOT / _rid)
    t1_wall += w
    t1_n += n
check("tier 1: 981 problem-arms", t1_n == 981, f"actual {t1_n}")
check("tier 1: wall clock binds once", t1_wall == 1, f"actual {t1_wall}")
check("tier 1: the one cut-off is the control",
      sum(_limit_hits(ROOT / r)[0] for k, r in t1["runs"].items() if not k.endswith("none")) == 0)

sweep_wall = {}
for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
    for p in discover(ROOT, bench, "vllm",
                      ["search.samples_per_step=32",
                       "policy_config.premise_free_fraction=0.25", f"n_problems={n}"]):
        arm = arm_of(load_draw(p))
        if arm not in ("li", "sv"):
            continue
        w, t = _limit_hits(p)
        a, b = sweep_wall.get(arm, (0, 0))
        sweep_wall[arm] = (a + w, b + t)
check("pass@8 li wall clock = 18 / 2,616", sweep_wall["li"] == (18, 2616), f"{sweep_wall['li']}")
check("pass@8 sv wall clock = 9 / 2,616", sweep_wall["sv"] == (9, 2616), f"{sweep_wall['sv']}")
check("pass@8 asymmetry is exactly 2x", sweep_wall["li"][0] == 2 * sweep_wall["sv"][0])

# --- the approximation late interaction is forced into, as section 11.9 constraint 1 -------------
_rec = D.table("li_recall_fate_m")["recall_by_n_candidates"]["50000"]
check("li recall@50k = 0.979", abs(_rec["recall"] - 0.9787) < 0.0005, f"{_rec['recall']:.4f}")
check("li: 17 of 141 queries still lossy at 50k", 141 - _rec["n_lossless"] == 17,
      f"actual {141 - _rec['n_lossless']}")

# --- the hypothesis structure: three primaries, seven secondaries --------------------------------
for h in ("**H1**", "**H2**", "**H3**"):
    says(f"primary hypothesis {h}", h)
for s in ("**S1**", "**S4**", "**S6**", "**S7**"):
    says(f"secondary finding {s}", s)
check("no fourth primary hypothesis", "**H4**" not in TXT)
check("no tenth primary hypothesis", "**H10**" not in TXT)

# --- figures referenced exist -------------------------------------------------------------------
refs = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", TXT)
check("17 figures referenced", len(refs) == 17, f"actual {len(refs)}")
for r in refs:
    check(f"exists {r}", pathlib.Path(r).exists())
nums = sorted(int(re.search(r"fig(\d+)_", r).group(1)) for r in refs)
check("figures 1..17 each referenced once", nums == list(range(1, 18)), f"{nums}")
order = [int(re.search(r"fig(\d+)_", r).group(1)) for r in refs]
check("figures appear in numerical order", order == sorted(order), f"{order}")
for p in sorted(pathlib.Path("figures").glob("*.png")):
    check(f"{p.name} is referenced", any(p.name in r for r in refs))

# --- claims that must NOT appear (stale numbers from superseded run sets) ------------------------
for banned, why in ((" 19/186", "pre-staging ProofNet control"),
                    ("96 → 96 → 85", "unsupported undecodable trend"),
                    ("257 → 218 → 196", "unsupported undecodable trend"),
                    ("+14 of 327`", "pre-staging retrieval effect"),
                    ("110.16", "pre-discount equal-budget figure"),
                    ("+3.16", "pre-discount equal-budget contrast")):
    check(f"no stale {banned!r} ({why})", banned not in TXT)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
