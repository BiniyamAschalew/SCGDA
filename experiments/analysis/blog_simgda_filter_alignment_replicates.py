import io
import multiprocessing as mp
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _jobs() -> list[dict]:
    return [
        {"seed": 1, "epochs": 220, "weight": 0.20, "beta": 0.0, "gpu": 0, "tag": "baseline_best"},
        {"seed": 2, "epochs": 220, "weight": 0.20, "beta": 0.0, "gpu": 1, "tag": "baseline_best"},
        {"seed": 1, "epochs": 220, "weight": 0.05, "beta": 0.05, "gpu": 2, "tag": "align_best_single"},
        {"seed": 2, "epochs": 220, "weight": 0.05, "beta": 0.05, "gpu": 3, "tag": "align_best_single"},
        {"seed": 1, "epochs": 300, "weight": 0.05, "beta": 0.10, "gpu": 4, "tag": "align_best_mean"},
        {"seed": 2, "epochs": 300, "weight": 0.05, "beta": 0.10, "gpu": 5, "tag": "align_best_mean"},
    ]


def _run_one(job: dict) -> dict:
    from  utils.config_utils import build_config
    from  run import run

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            config = build_config(
                {"data": "blog", "expt": "default", "model": "simgda_filter"},
                {
                    "expt": {
                        "source": "Blog1",
                        "target": "Blog2",
                        "device": f"cuda:{job['gpu']}",
                        "seed": job["seed"],
                        "verbose": 0,
                        "wandb_enabled": False,
                        "save_model": False,
                    },
                    "model": {
                        "epochs": job["epochs"],
                        "weight": job["weight"],
                        "beta": job["beta"],
                        "adv": False,
                    },
                },
            )
            result = run(config)
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "train_time": float("nan"),
        }
    result.update(job)
    return result


def main():
    jobs = _jobs()
    out_dir = os.path.join(ROOT_DIR, "__saved__/results/evaluation")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%m%d%H%M%S")
    raw_path = os.path.join(out_dir, f"{stamp}_blog_simgda_filter_alignment_replicates.csv")
    summary_path = os.path.join(out_dir, f"{stamp}_blog_simgda_filter_alignment_replicates_summary.csv")

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).to_csv(raw_path, index=False)
            print(
                f"[{idx}/{len(jobs)}] "
                f"{result['tag']} seed={result['seed']} "
                f"beta={result['beta']:.2f} weight={result['weight']:.2f} "
                f"epochs={result['epochs']} micro_f1={result['micro_f1']:.6f}"
            )

    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy()
    summary = (
        ok.groupby(["tag", "beta", "weight", "epochs"], as_index=False)[["micro_f1", "macro_f1", "train_time"]]
        .agg(
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            train_time_mean=("train_time", "mean"),
            runs=("micro_f1", "count"),
        )
        .sort_values("micro_f1_mean", ascending=False)
    )
    summary.to_csv(summary_path, index=False)

    print(f"raw_path={raw_path}")
    print(f"summary_path={summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
