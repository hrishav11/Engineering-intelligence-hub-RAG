"""Eval runner: for each question × method, run retrieval + answer + judge,
aggregate, save JSON, print scorecard.

Multi-run mode (`runs > 1`) re-runs the full pipeline N times to surface the
real variance from non-deterministic components (HyDE, agentic, LLM judge).
The aggregation reports mean ± stddev so we can tell whether observed method
differences are bigger than the noise floor."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..answer import ask as ask_fn
from . import judge as judge_mod
from . import metrics as metrics_mod


@dataclass
class QuestionResult:
    method: str
    qid: str
    question: str
    category: str
    run: int                      # 0-indexed run number for multi-run averaging
    hit_at_1: int                 # strict: path AND symbol
    hit_at_3: int
    hit_at_10: int
    mrr: float
    num_hits_in_topk: int
    p_hit_at_1: int               # path-only: did the right FILE appear?
    p_hit_at_3: int
    p_hit_at_10: int
    p_mrr: float
    path_coverage: float          # coverage anywhere in cumulative hits
    symbol_coverage: float
    judge_score: int | None
    judge_reason: str | None
    answer: str
    hit_paths: list[str]


def load_eval_set(path: Path) -> list[dict]:
    with path.open() as f:
        return yaml.safe_load(f)


def run_eval(
    questions: list[dict],
    methods: list[str],
    k: int = 10,
    skip_judge: bool = False,
    runs: int = 1,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    total = len(questions) * len(methods) * runs
    i = 0
    for run_idx in range(runs):
        if runs > 1:
            print(f"\n=== run {run_idx + 1}/{runs} ===")
        for q in questions:
            for method in methods:
                i += 1
                print(f"[{i}/{total}] r{run_idx} {method:>6} · {q['id']}")
                try:
                    got = ask_fn(q["question"], k=k, method=method)
                except Exception as e:
                    print(f"   ERROR: {e}")
                    continue
                ret_score = metrics_mod.score(
                    got["hits"], q["expected_paths"], q.get("expected_symbols"),
                )
                judged: judge_mod.Judgment | None = None
                if not skip_judge:
                    try:
                        judged = judge_mod.judge(q["question"], got["answer"])
                    except Exception as e:
                        print(f"   judge error: {e}")
                results.append(QuestionResult(
                    method=method,
                    qid=q["id"],
                    question=q["question"],
                    category=q["category"],
                    run=run_idx,
                    hit_at_1=ret_score.hit_at_1,
                    hit_at_3=ret_score.hit_at_3,
                    hit_at_10=ret_score.hit_at_10,
                    mrr=ret_score.mrr,
                    num_hits_in_topk=ret_score.num_hits_in_topk,
                    p_hit_at_1=ret_score.p_hit_at_1,
                    p_hit_at_3=ret_score.p_hit_at_3,
                    p_hit_at_10=ret_score.p_hit_at_10,
                    p_mrr=ret_score.p_mrr,
                    path_coverage=ret_score.path_coverage,
                    symbol_coverage=ret_score.symbol_coverage,
                    judge_score=judged.score if judged else None,
                    judge_reason=judged.reason if judged else None,
                    answer=got["answer"],
                    hit_paths=[h["meta"].get("source_path", "") for h in got["hits"]],
                ))
    return results


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def aggregate(results: list[QuestionResult]) -> dict[str, dict[str, Any]]:
    """Aggregate per method. With runs>1, returns 'mean ± stddev' style metrics
    computed across per-run means (so the stddev reflects run-to-run variance)."""
    by_method: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for method, rs in by_method.items():
        # Bucket per run, compute per-run means, then aggregate
        by_run: dict[int, list[QuestionResult]] = {}
        for r in rs:
            by_run.setdefault(r.run, []).append(r)
        n_runs = len(by_run)
        n_questions = len(rs) // max(n_runs, 1)

        def metric_mean_std(extract):
            run_means = []
            for run_rs in by_run.values():
                vals = [extract(r) for r in run_rs if extract(r) is not None]
                if vals:
                    run_means.append(sum(vals) / len(vals))
            return _mean_std(run_means)

        # Strict (path+symbol) metrics
        h1_m, h1_s = metric_mean_std(lambda r: r.hit_at_1)
        h3_m, h3_s = metric_mean_std(lambda r: r.hit_at_3)
        h10_m, h10_s = metric_mean_std(lambda r: r.hit_at_10)
        mrr_m, mrr_s = metric_mean_std(lambda r: r.mrr)
        # Path-only metrics (looser, closer to user experience)
        ph1_m, ph1_s = metric_mean_std(lambda r: getattr(r, "p_hit_at_1", 0))
        ph3_m, ph3_s = metric_mean_std(lambda r: getattr(r, "p_hit_at_3", 0))
        ph10_m, ph10_s = metric_mean_std(lambda r: getattr(r, "p_hit_at_10", 0))
        pmrr_m, pmrr_s = metric_mean_std(lambda r: getattr(r, "p_mrr", 0.0))
        pc_m, pc_s = metric_mean_std(lambda r: r.path_coverage)
        sc_m, sc_s = metric_mean_std(lambda r: r.symbol_coverage)
        j_m, j_s = metric_mean_std(lambda r: r.judge_score)

        out[method] = {
            "n_questions": n_questions,
            "n_runs": n_runs,
            "hit@1": (round(h1_m, 3), round(h1_s, 3)),
            "hit@3": (round(h3_m, 3), round(h3_s, 3)),
            "hit@10": (round(h10_m, 3), round(h10_s, 3)),
            "mrr": (round(mrr_m, 3), round(mrr_s, 3)),
            "p_hit@1": (round(ph1_m, 3), round(ph1_s, 3)),
            "p_hit@3": (round(ph3_m, 3), round(ph3_s, 3)),
            "p_hit@10": (round(ph10_m, 3), round(ph10_s, 3)),
            "p_mrr": (round(pmrr_m, 3), round(pmrr_s, 3)),
            "path_cov": (round(pc_m, 3), round(pc_s, 3)),
            "symbol_cov": (round(sc_m, 3), round(sc_s, 3)),
            "judge_avg": (round(j_m, 2), round(j_s, 2)) if j_m else None,
        }
    return out


def aggregate_by_category(results: list[QuestionResult]) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {method: {category: {metric: mean across all runs}}}. Stddev is in the
    top-level aggregate; category breakdown stays as plain means for readability."""
    grouped: dict[str, dict[str, list[QuestionResult]]] = {}
    for r in results:
        grouped.setdefault(r.method, {}).setdefault(r.category, []).append(r)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for method, cats in grouped.items():
        out[method] = {}
        for cat, rs in cats.items():
            n = len(rs)
            n_runs = len({r.run for r in rs}) or 1
            n_questions = n // n_runs
            out[method][cat] = {
                "n": n_questions,
                "hit@3": round(sum(r.hit_at_3 for r in rs) / n, 3),
                "mrr": round(sum(r.mrr for r in rs) / n, 3),
                "path_cov": round(sum(r.path_coverage for r in rs) / n, 3),
            }
    return out


def save_results(results: list[QuestionResult], summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
