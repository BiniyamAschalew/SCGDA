"""Merge benchmark-with-std CSVs and render a LaTeX benchmark table."""

import argparse
import re
from pathlib import Path

import pandas as pd


META_COLS = ["dataset", "source", "target", "scenario"]
PREFERRED_MODEL_ORDER = [
    "a2gnn",
    "adagcn",
    "dane",
    "dgsda",
    "grade",
    "jhgda",
    "kbl",
    "pairalign",
    "specreg",
    "strurw",
    "tdss",
    "udagcn",
]
OUR_MODELS = ["opal"]
DISPLAY_NAMES = {
    "a2gnn": "A2GNN",
    "adagcn": "AdaGCN",
    "dane": "DANE",
    "dgsda": "DGSDA",
    "grade": "GRADE",
    "jhgda": "JHGDA",
    "kbl": "KBL",
    "opal": "OPAL",
    "pairalign": "PairAlign",
    "specreg": "SpecReg",
    "strurw": "StruRW",
    "tdss": "TDSS",
    "udagcn": "UDAGCN",
}
SECTION_SPECS = [
    {
        "groups": [
            (
                "Airport",
                [
                    (("Airport", "BRAZIL", "EUROPE"), "B$\\to$E"),
                    (("Airport", "BRAZIL", "USA"), "B$\\to$U"),
                    (("Airport", "EUROPE", "BRAZIL"), "E$\\to$B"),
                    (("Airport", "EUROPE", "USA"), "E$\\to$U"),
                    (("Airport", "USA", "BRAZIL"), "U$\\to$B"),
                    (("Airport", "USA", "EUROPE"), "U$\\to$E"),
                ],
            ),
            (
                "Blog",
                [
                    (("Blog", "Blog1", "Blog2"), "B1$\\to$B2"),
                    (("Blog", "Blog2", "Blog1"), "B2$\\to$B1"),
                ],
            ),
        ]
    },
    {
        "groups": [
            (
                "Citation",
                [
                    (("Citation", "ACMv9", "Citationv1"), "A$\\to$C"),
                    (("Citation", "ACMv9", "DBLPv7"), "A$\\to$D"),
                    (("Citation", "Citationv1", "ACMv9"), "C$\\to$A"),
                    (("Citation", "Citationv1", "DBLPv7"), "C$\\to$D"),
                    (("Citation", "DBLPv7", "ACMv9"), "D$\\to$A"),
                    (("Citation", "DBLPv7", "Citationv1"), "D$\\to$C"),
                ],
            ),
            (
                "Twitch",
                [
                    (("Twitch", "DE", "EN"), "DE$\\to$EN"),
                    (("Twitch", "EN", "DE"), "EN$\\to$DE"),
                ],
            ),
        ]
    },
]
DEFAULT_CAPTION = (
    "Micro-F1 (\\%) on Airport, Blog, Citation, and Twitch datasets. "
    "Each entry reports mean and standard deviation across seeds. "
    "\\first{Red}, \\second{Orange}, and \\third{Blue} indicate the top 3 "
    "results; \\textbf{bold} indicates the 4th-best result. Avg. Rank is "
    "computed over all available transfer tasks, with lower being better."
)
DEFAULT_LABEL = "table:benchmark_full_std_merged"
VALUE_PATTERN = re.compile(
    r"^\s*([0-9]*\.?[0-9]+)(?:\s*(?:±|\\pm|\+/-)\s*([0-9]*\.?[0-9]+))?\s*$"
)


def model_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLS]


def load_benchmark_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in META_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.copy()


def merge_benchmark_csvs(paths: list[Path]) -> pd.DataFrame:
    merged = None
    for path in paths:
        current = load_benchmark_csv(path)
        if merged is None:
            merged = current
            continue

        overlap = sorted(set(model_columns(merged)) & set(model_columns(current)))
        merged = merged.merge(current, on=META_COLS, how="outer", suffixes=("_left", "_right"))

        for model in overlap:
            left_col = f"{model}_left"
            right_col = f"{model}_right"
            if left_col not in merged.columns or right_col not in merged.columns:
                continue

            left = merged[left_col]
            right = merged[right_col]
            left_text = left.fillna("").astype(str).str.strip()
            right_text = right.fillna("").astype(str).str.strip()
            conflicts = (left_text != "") & (right_text != "") & (left_text != right_text)
            if conflicts.any():
                scenario = merged.loc[conflicts, "scenario"].iloc[0]
                raise ValueError(
                    f"Conflicting values found for model '{model}' in scenario '{scenario}'."
                )

            merged[model] = left.where(left.notna(), right)
            merged = merged.drop(columns=[left_col, right_col])

    if merged is None:
        raise ValueError("No input CSV files were provided.")

    merged = merged.sort_values(META_COLS).reset_index(drop=True)
    ordered_models = order_models(model_columns(merged))
    return merged[META_COLS + ordered_models]


def order_models(models: list[str]) -> list[str]:
    ours = [model for model in OUR_MODELS if model in models]
    remaining = sorted(
        model for model in models if model not in PREFERRED_MODEL_ORDER and model not in ours
    )
    preferred = [model for model in PREFERRED_MODEL_ORDER if model in models]
    return preferred + remaining + ours


def format_model_name(model: str) -> str:
    display = DISPLAY_NAMES.get(model, model.upper())
    if model in OUR_MODELS:
        return rf"\entryfmt{{\textbf{{{display}}} (ours)}}"
    return rf"\entryfmt{{{display}}}"


def parse_metric(cell: object) -> tuple[float | None, float | None]:
    if pd.isna(cell):
        return None, None

    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return None, None

    match = VALUE_PATTERN.match(text)
    if not match:
        raise ValueError(f"Could not parse metric cell: {text!r}")

    mean = float(match.group(1))
    std = float(match.group(2)) if match.group(2) is not None else None
    return mean, std


def build_metric_frames(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = merged.set_index(META_COLS)
    models = model_columns(merged)

    mean_df = pd.DataFrame(index=indexed.index)
    std_df = pd.DataFrame(index=indexed.index)

    for model in models:
        parsed = indexed[model].apply(parse_metric)
        mean_df[model] = parsed.apply(lambda item: item[0])
        std_df[model] = parsed.apply(lambda item: item[1])

    return mean_df, std_df


def validate_known_scenarios(mean_df: pd.DataFrame) -> None:
    known = {
        scenario_key
        for section in SECTION_SPECS
        for _, scenarios in section["groups"]
        for scenario_key, _ in scenarios
    }
    actual = set(mean_df.index.droplevel("scenario").tolist())
    extra = sorted(actual - known)
    if extra:
        raise ValueError(f"Unhandled scenarios found in merged benchmark data: {extra}")


def format_result(mean: float | None, std: float | None, rank: float | None) -> str:
    if mean is None or pd.isna(mean):
        return r"\entryfmt{-}"

    content = rf"\entryfmt{{{mean * 100:.2f}}}"
    if std is not None and not pd.isna(std):
        content += rf" {{\stdfmt{{$\pm$ {std * 100:.2f}}}}}"

    if rank is None or pd.isna(rank):
        return content
    if abs(rank - 1.0) < 1e-9:
        return rf"\first{{{content}}}"
    if abs(rank - 2.0) < 1e-9:
        return rf"\second{{{content}}}"
    if abs(rank - 3.0) < 1e-9:
        return rf"\third{{{content}}}"
    if abs(rank - 4.0) < 1e-9:
        return rf"\textbf{{{content}}}"
    return content


def format_avg_rank(avg_rank: float | None) -> str:
    if avg_rank is None or pd.isna(avg_rank):
        return r"\entryfmt{-}"
    return rf"\entryfmt{{{avg_rank:.2f}}}"


def render_section(
    section: dict,
    models: list[str],
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    avg_rank: pd.Series,
) -> list[str]:
    scenario_lookup = {index[:-1]: index for index in mean_df.index}
    headers = []
    cmidrules = []
    scenario_columns = []
    col_start = 2

    for group_index, (dataset_name, scenarios) in enumerate(section["groups"]):
        align = "c|" if group_index == len(section["groups"]) - 1 else "c"
        headers.append(
            rf"\multicolumn{{{len(scenarios)}}}{{{align}}}{{\headerfmt{{\textbf{{{dataset_name}}}}}}}"
        )
        cmidrules.append(rf"\cmidrule(lr){{{col_start}-{col_start + len(scenarios) - 1}}}")
        col_start += len(scenarios)
        scenario_columns.extend(scenarios)

    lines = [
        r"\multirow{2}{*}{\headerfmt{\textbf{Models}}} & "
        + " & ".join(headers)
        + r" & \multirow{2}{*}{\headerfmt{\textbf{Avg. Rank}}} \\",
        " ".join(cmidrules),
        "& "
        + " & ".join(rf"\headerfmt{{{label}}}" for _, label in scenario_columns)
        + r" & \\",
        r"\midrule",
    ]

    for model in models:
        row = [format_model_name(model)]
        for scenario_key, _ in scenario_columns:
            row_key = scenario_lookup.get(scenario_key)
            if row_key is not None:
                mean = mean_df.at[row_key, model] if model in mean_df.columns else None
                std = std_df.at[row_key, model] if model in std_df.columns else None
                rank = rank_df.at[row_key, model] if model in rank_df.columns else None
                row.append(format_result(mean, std, rank))
                continue
            row.append(r"\entryfmt{-}")

        row.append(format_avg_rank(avg_rank.get(model)))
        lines.append(" & ".join(row) + r" \\")

    return lines


def render_latex_table(merged: pd.DataFrame, caption: str, label: str) -> str:
    models = model_columns(merged)
    mean_df, std_df = build_metric_frames(merged)
    validate_known_scenarios(mean_df)

    rank_df = mean_df.rank(axis=1, method="average", ascending=False)
    avg_rank = rank_df.mean(axis=0, skipna=True)

    sections = []
    for section in SECTION_SPECS:
        sections.append(render_section(section, models, mean_df, std_df, rank_df, avg_rank))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\newcommand{\headersize}{\normalsize}",
        r"\newcommand{\entrysize}{\normalsize}",
        r"\newcommand{\stdsize}{\scriptsize}",
        r"\newcommand{\headerfmt}[1]{{\headersize #1}}",
        r"\newcommand{\entryfmt}[1]{{\entrysize #1}}",
        r"\newcommand{\stdfmt}[1]{{\stdsize #1}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc|c}",
        r"\toprule",
    ]

    lines.extend(sections[0])
    lines.append(r"\midrule")
    lines.extend(sections[1])
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge benchmark-with-std CSV files and render a LaTeX table."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, help="Input benchmark_*_with_std.csv files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--output-name",
        type=str,
        default="benchmark_micro_f1",
        help="Base filename for the merged CSV and generated LaTeX table.",
    )
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION)
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL)
    args = parser.parse_args()

    merged = merge_benchmark_csvs(args.inputs)
    table = render_latex_table(merged, caption=args.caption, label=args.label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = args.output_dir / f"{args.output_name}_with_std.csv"
    table_path = args.output_dir / f"{args.output_name}_table_with_std.tex"

    merged.to_csv(merged_csv_path, index=False)
    table_path.write_text(table, encoding="utf-8")

    print(f"Saved merged CSV: {merged_csv_path}")
    print(f"Saved LaTeX table: {table_path}")


if __name__ == "__main__":
    main()
