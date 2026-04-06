from  experiments.analysis.bound_importance.experiment import main

main()

# """
# Create a LaTeX benchmark table from multiple CSV result files.
# The table includes dataset super-columns, transfer scenarios, and
# mean +/- std entries with best/second-best highlighting per scenario.
# """

# import argparse
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# import pandas as pd


# DATASET_ORDER = ["Airport", "Blog", "Citation"]
# DATASET_DISPLAY = {
#     "Airport": "Airport",
#     "Blog": "Blog",
#     "Citation": "Citation",  # change to "ArnetMiner" if desired
# }

# DOMAIN_CANONICAL = {
#     "europe": "Europe",
#     "usa": "USA",
#     "blog1": "Blog1",
#     "blog2": "Blog2",
#     "citationv1": "Citationv1",
#     "dblpv7": "DBLPv7",
#     "acmv9": "ACMv9",
# }

# DOMAIN_ABBREV = {
#     "Europe": "E",
#     "USA": "U",
#     "Blog1": "B1",
#     "Blog2": "B2",
#     "Citationv1": "C",
#     "DBLPv7": "D",
#     "ACMv9": "A",
# }

# MODEL_DISPLAY = {
#     "a2gnn": "A2GNN",
#     "acdne": "ACDNE",
#     "adagcn": "AdaGCN",
#     "asn": "ASN",
#     "gnn": "GNN",
#     "simgda": "SimGDA",
# }

# MODEL_ORDER = ["A2GNN", "ACDNE", "AdaGCN", "ASN", "GNN", "SimGDA"]

# SCENARIO_ORDER = {
#     "Airport": [("Europe", "USA"), ("USA", "Europe")],
#     "Blog": [("Blog1", "Blog2"), ("Blog2", "Blog1")],
#     "Citation": [("Citationv1", "ACMv9"), ("DBLPv7", "ACMv9")],
# }


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Build a LaTeX benchmark table from CSV result files."
#     )
#     parser.add_argument(
#         "--results-dir",
#         default=None,
#         help="Directory containing benchmark CSV files (e.g., ./__saved__/results/benchmark/run_0125_123456). If not specified, uses the latest run directory.",
#     )
#     parser.add_argument(
#         "--pattern",
#         default="benchmark_*_seed*.csv",
#         help="Glob pattern for CSV files in results-dir (default matches seed-based files).",
#     )
#     parser.add_argument(
#         "--files",
#         nargs="*",
#         default=None,
#         help="Explicit list of CSV files to use (overrides results-dir/pattern).",
#     )
#     parser.add_argument(
#         "--metric",
#         default="micro_f1",
#         help="Metric column name to aggregate (e.g., micro_f1 or macro_f1).",
#     )
#     parser.add_argument(
#         "--output",
#         default="./__saved__/results/benchmark/benchmark_results9.tex",
#         help="Path to save the LaTeX table (default: saves in the run directory).",
#     )
#     parser.add_argument(
#         "--caption",
#         default=None,
#         help="LaTeX caption for the table (auto-generated if omitted).",
#     )
#     parser.add_argument(
#         "--label",
#         default="tab:benchmark_results",
#         help="LaTeX label for the table.",
#     )
#     parser.add_argument(
#         "--round",
#         type=int,
#         default=2,
#         help="Decimal places for mean/std formatting and ranking.",
#     )
#     return parser.parse_args()


# def resolve_paths(
#     results_dir: Optional[str], pattern: str, files: Optional[Sequence[str]]
# ) -> Tuple[List[Path], Path]:
#     if files:
#         paths = [Path(p) for p in files]
#         # Use parent of first file as run_dir
#         run_dir = paths[0].parent if paths else Path("./__saved__/results/benchmark")
#     else:
#         if results_dir is None:
#             # Find the latest run directory
#             benchmark_base = Path("./__saved__/results/benchmark")
#             run_dirs = sorted(benchmark_base.glob("run_*"), reverse=True)
#             if not run_dirs:
#                 raise FileNotFoundError("No run directories found in ./__saved__/results/benchmark/")
#             results_dir = str(run_dirs[0])
#             print(f"Using latest run directory: {results_dir}")
        
#         run_dir = Path(results_dir)
#         paths = sorted(run_dir.glob(pattern))

#     if not paths:
#         raise FileNotFoundError("No benchmark CSV files found.")

#     missing = [p for p in paths if not p.exists()]
#     if missing:
#         missing_str = ", ".join(str(p) for p in missing)
#         raise FileNotFoundError(f"Missing CSV files: {missing_str}")

#     return paths, run_dir


# def load_results(paths: Iterable[Path]) -> pd.DataFrame:
#     frames = [pd.read_csv(p) for p in paths]
#     return pd.concat(frames, ignore_index=True)


# def normalize_dataset(name: str) -> str:
#     key = name.strip()
#     lookup = key.lower()
#     for canonical in DATASET_DISPLAY:
#         if canonical.lower() == lookup:
#             return canonical
#     return key


# def normalize_domain(name: str) -> str:
#     key = name.strip()
#     canonical = DOMAIN_CANONICAL.get(key.lower())
#     return canonical if canonical else key


# def normalize_model(name: str) -> Tuple[str, str]:
#     key = name.strip().lower()
#     display = MODEL_DISPLAY.get(key, name.strip())
#     return key, display


# def abbreviate_domain(name: str) -> str:
#     return DOMAIN_ABBREV.get(name, name)


# def format_metric_name(metric: str) -> str:
#     parts = metric.replace("_", " ").split()
#     return "-".join(p.capitalize() for p in parts)


# def dataset_list_phrase(datasets: Sequence[str]) -> str:
#     if not datasets:
#         return ""
#     if len(datasets) == 1:
#         return datasets[0]
#     if len(datasets) == 2:
#         return f"{datasets[0]} and {datasets[1]}"
#     return f"{', '.join(datasets[:-1])}, and {datasets[-1]}"


# def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["dataset"] = df["dataset"].astype(str).str.strip().apply(normalize_dataset)
#     df["source"] = df["source"].astype(str).str.strip().apply(normalize_domain)
#     df["target"] = df["target"].astype(str).str.strip().apply(normalize_domain)

#     model_keys: List[str] = []
#     model_displays: List[str] = []
#     for name in df["model"].astype(str).tolist():
#         key, display = normalize_model(name)
#         model_keys.append(key)
#         model_displays.append(display)
#     df["model_key"] = model_keys
#     df["model_display"] = model_displays
#     return df


# def compute_stats(
#     df: pd.DataFrame, metric: str
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
#     if metric not in df.columns:
#         raise KeyError(f"Metric '{metric}' not found in columns: {df.columns.tolist()}")

#     metric_values = df[metric].astype(float)
#     scale = 100.0 if metric_values.max() <= 1.0 else 1.0
#     df = df.copy()
#     df[metric] = metric_values * scale

#     grouped = df.groupby(["model_display", "dataset", "source", "target"])[metric]
#     stats = grouped.agg(["mean", "std", "count"]).reset_index()
#     stats["std"] = stats["std"].fillna(0.0)

#     mean_df = stats.pivot_table(
#         index="model_display",
#         columns=["dataset", "source", "target"],
#         values="mean",
#     )
#     std_df = stats.pivot_table(
#         index="model_display",
#         columns=["dataset", "source", "target"],
#         values="std",
#     )
#     count_df = stats.pivot_table(
#         index="model_display",
#         columns=["dataset", "source", "target"],
#         values="count",
#     )

#     return mean_df, std_df, count_df, scale


# def resolve_dataset_order(present: Sequence[str]) -> List[str]:
#     order = [d for d in DATASET_ORDER if d in present]
#     extras = sorted(d for d in present if d not in order)
#     return order + extras


# def resolve_scenario_order(
#     dataset: str, available_pairs: Sequence[Tuple[str, str]]
# ) -> List[Tuple[str, str]]:
#     if dataset in SCENARIO_ORDER:
#         ordered = [p for p in SCENARIO_ORDER[dataset] if p in available_pairs]
#         extras = sorted(p for p in available_pairs if p not in ordered)
#         return ordered + extras
#     return sorted(available_pairs)


# def build_column_plan(
#     mean_df: pd.DataFrame,
# ) -> Tuple[List[Tuple[str, str, str]], List[str], List[Tuple[str, int, int]]]:
#     if not isinstance(mean_df.columns, pd.MultiIndex):
#         raise ValueError("Expected MultiIndex columns for mean_df.")

#     present_datasets = sorted({c[0] for c in mean_df.columns})
#     dataset_order = resolve_dataset_order(present_datasets)

#     columns: List[Tuple[str, str, str]] = []
#     header_labels: List[str] = []
#     dataset_spans: List[Tuple[str, int, int]] = []
#     col_idx = 2  # LaTeX columns start at 1; column 1 is the model name.

#     for dataset in dataset_order:
#         available_pairs = sorted({(c[1], c[2]) for c in mean_df.columns if c[0] == dataset})
#         scenario_order = resolve_scenario_order(dataset, available_pairs)
#         if not scenario_order:
#             continue

#         start = col_idx
#         for source, target in scenario_order:
#             columns.append((dataset, source, target))
#             label = f"{abbreviate_domain(source)} $\\\\to$ {abbreviate_domain(target)}"
#             header_labels.append(label)
#             col_idx += 1
#         end = col_idx - 1
#         dataset_spans.append((DATASET_DISPLAY.get(dataset, dataset), start, end))

#     return columns, header_labels, dataset_spans


# def order_models(df: pd.DataFrame) -> pd.DataFrame:
#     order = [m for m in MODEL_ORDER if m in df.index]
#     extras = sorted(m for m in df.index if m not in order)
#     return df.reindex(order + extras)


# def compute_highlights(
#     mean_df: pd.DataFrame, round_digits: int
# ) -> Dict[Tuple[str, Tuple[str, str, str]], str]:
#     highlights: Dict[Tuple[str, Tuple[str, str, str]], str] = {}
#     for col in mean_df.columns:
#         series = mean_df[col]
#         ranked = series.round(round_digits)
#         valid = ranked.dropna()
#         if valid.empty:
#             continue

#         best_val = valid.max()
#         best_models = valid[valid == best_val].index.tolist()
#         second_candidates = valid[valid < best_val]
#         second_models: List[str] = []
#         if not second_candidates.empty:
#             second_val = second_candidates.max()
#             second_models = second_candidates[second_candidates == second_val].index.tolist()

#         for model in best_models:
#             highlights[(model, col)] = "best"
#         for model in second_models:
#             highlights[(model, col)] = "second"
#     return highlights


# def format_cell(
#     mean: Optional[float],
#     std: Optional[float],
#     highlight: Optional[str],
#     round_digits: int,
# ) -> str:
#     if mean is None or pd.isna(mean):
#         return "--"
#     mean_str = f"{mean:.{round_digits}f}"
#     std_str = f"{0.0 if std is None or pd.isna(std) else std:.{round_digits}f}"
#     if highlight == "best":
#         mean_str = f"\\\\textbf{{{mean_str}}}"
#     elif highlight == "second":
#         mean_str = f"\\\\underline{{{mean_str}}}"
#     return f"{mean_str} $\\\\pm$ {std_str}"


# def build_table_values(
#     mean_df: pd.DataFrame,
#     std_df: pd.DataFrame,
#     round_digits: int,
# ) -> pd.DataFrame:
#     highlights = compute_highlights(mean_df, round_digits)
#     table_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
#     for model in mean_df.index:
#         for col in mean_df.columns:
#             mean = mean_df.loc[model, col]
#             std = std_df.loc[model, col] if col in std_df.columns else None
#             highlight = highlights.get((model, col))
#             table_df.loc[model, col] = format_cell(mean, std, highlight, round_digits)
#     return table_df


# def warn_on_counts(count_df: pd.DataFrame, min_count: int = 2) -> None:
#     if count_df.empty:
#         return
#     low_entries: List[Tuple[str, str, str, str, int]] = []
#     for model in count_df.index:
#         for dataset, source, target in count_df.columns:
#             count = count_df.loc[model, (dataset, source, target)]
#             if pd.notna(count) and count < min_count:
#                 low_entries.append((model, dataset, source, target, int(count)))
#     if low_entries:
#         print("Warning: low sample counts detected for some entries:")
#         for model, dataset, source, target, count in low_entries:
#             print(f"  {model} | {dataset} {source}->{target}: n={count}")


# def build_latex_table(
#     table_df: pd.DataFrame,
#     header_labels: Sequence[str],
#     dataset_spans: Sequence[Tuple[str, int, int]],
#     caption: str,
#     label: str,
# ) -> str:
#     ncols = len(header_labels)
#     if ncols == 0:
#         raise ValueError("No columns available to build the LaTeX table.")

#     col_spec = "l" + "c" * ncols
#     lines: List[str] = []
#     lines.append("\\\\begin{table*}[t]")
#     lines.append("\\\\centering")
#     lines.append(f"\\\\caption{{{caption}}}")
#     lines.append(f"\\\\label{{{label}}}")
#     lines.append(f"\\\\begin{{tabular}}{{{col_spec}}}")
#     lines.append("\\\\toprule")

#     header1_parts = ["\\\\multirow{2}{*}{\\\\textbf{Models}}"]
#     for name, start, end in dataset_spans:
#         span = end - start + 1
#         header1_parts.append(f"\\\\multicolumn{{{span}}}{{c}}{{\\\\textbf{{{name}}}}}")
#     lines.append(" & ".join(header1_parts) + " \\\\")

#     cmid_parts = [f"\\\\cmidrule(lr){{{start}-{end}}}" for _, start, end in dataset_spans]
#     lines.append(" ".join(cmid_parts))

#     header2 = " & ".join([""] + list(header_labels)) + " \\\\"
#     lines.append(header2)
#     lines.append("\\\\midrule")

#     for model in table_df.index:
#         row_vals = [model] + table_df.loc[model].tolist()
#         lines.append(" & ".join(row_vals) + " \\\\")

#     lines.append("\\\\bottomrule")
#     lines.append("\\\\end{tabular}")
#     lines.append("\\\\end{table*}")
#     return "\n".join(lines)


# def main() -> None:
#     args = parse_args()
#     paths, run_dir = resolve_paths(args.results_dir, args.pattern, args.files)
#     df = load_results(paths)
#     df = normalize_results(df)

#     mean_df, std_df, count_df, scale = compute_stats(df, args.metric)
#     warn_on_counts(count_df)

#     columns, header_labels, dataset_spans = build_column_plan(mean_df)
#     mean_df = mean_df.reindex(columns=pd.MultiIndex.from_tuples(columns))
#     std_df = std_df.reindex(columns=pd.MultiIndex.from_tuples(columns))

#     mean_df = order_models(mean_df)
#     std_df = std_df.reindex(mean_df.index)

#     table_df = build_table_values(mean_df, std_df, args.round)

#     datasets_display = [name for name, _, _ in dataset_spans]
#     metric_name = format_metric_name(args.metric)
#     percent_suffix = " (\\\\%)" if scale == 100.0 else ""
#     if args.caption:
#         caption = args.caption
#     else:
#         caption = (
#             f"{metric_name}{percent_suffix} on "
#             f"{dataset_list_phrase(datasets_display)} datasets. "
#             "The best results are highlighted in \\\\textbf{bold} "
#             "and the second best are \\\\underline{underlined}."
#         )

#     latex = build_latex_table(
#         table_df=table_df,
#         header_labels=header_labels,
#         dataset_spans=dataset_spans,
#         caption=caption,
#         label=args.label,
#     )

#     # Use run_dir for output if default output path
#     if args.output == "./__saved__/results/benchmark/benchmark_results.tex":
#         output_path = run_dir / "benchmark_results.tex"
#     else:
#         output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.write_text(latex)
#     print(f"LaTeX table saved to {output_path}")


# if __name__ == "__main__":
#     main()
