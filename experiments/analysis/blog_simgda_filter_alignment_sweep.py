import itertools
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


def _run_one(job: dict) -> dict:
    from utils.config_utils import build_config
    from run import run

    device = f"cuda:{job['gpu']}"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        config = build_config(
            {"data": "blog", "expt": "default", "model": "simgda_filter"},
            {
                "expt": {
                    "source": "Blog1",
                    "target": "Blog2",
                    "device": device,
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
    result.update(
        {
            "epochs": job["epochs"],
            "weight": job["weight"],
            "beta": job["beta"],
            "gpu": job["gpu"],
        }
    )
    return result


def _job_grid(gpus: list[int]) -> list[dict]:
    seeds = [0]
    epochs_list = [220, 300]
    weights = [0.05, 0.1, 0.2]
    betas = [0.0, 0.05, 0.1]

    jobs = []
    for idx, (seed, epochs, weight, beta) in enumerate(
        itertools.product(seeds, epochs_list, weights, betas)
    ):
        jobs.append(
            {
                "seed": seed,
                "epochs": epochs,
                "weight": weight,
                "beta": beta,
                "gpu": gpus[idx % len(gpus)],
            }
        )
    return jobs


def _summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = (
        ok.groupby(["beta", "weight", "epochs"], as_index=False)[
            ["micro_f1", "macro_f1", "train_time"]
        ]
        .mean()
        .sort_values(["micro_f1", "macro_f1"], ascending=False)
    )

    beta_view = (
        ok.groupby(["beta"], as_index=False)[["micro_f1", "macro_f1", "train_time"]]
        .agg(
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            train_time_mean=("train_time", "mean"),
            runs=("micro_f1", "count"),
        )
        .sort_values("beta")
    )
    return grouped, beta_view


def main():
    gpus = list(range(8))
    jobs = _job_grid(gpus)

    out_dir = "./__saved__/results/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%m%d%H%M%S")
    raw_path = os.path.join(out_dir, f"{stamp}_blog_simgda_filter_alignment_raw.csv")
    summary_path = os.path.join(out_dir, f"{stamp}_blog_simgda_filter_alignment_summary.csv")
    beta_path = os.path.join(out_dir, f"{stamp}_blog_simgda_filter_alignment_beta.csv")

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=ctx) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).to_csv(raw_path, index=False)
            print(
                f"[{idx}/{len(jobs)}] "
                f"beta={result['beta']:.2f} weight={result['weight']:.2f} "
                f"epochs={result['epochs']} seed={result['seed']} "
                f"status={result['status']} micro_f1={result.get('micro_f1')}"
            )

    df = pd.DataFrame(results)
    grouped, beta_view = _summaries(df)
    grouped.to_csv(summary_path, index=False)
    beta_view.to_csv(beta_path, index=False)

    print(f"raw_path={raw_path}")
    print(f"summary_path={summary_path}")
    print(f"beta_path={beta_path}")
    if not grouped.empty:
        print("\nTop configurations:")
        print(grouped.head(10).to_string(index=False))
    if not beta_view.empty:
        print("\nBeta comparison:")
        print(beta_view.to_string(index=False))


if __name__ == "__main__":
    main()
