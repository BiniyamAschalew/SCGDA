"""
Let's load the benchmark result files, different csv for different seeds and then create a unified result table
Let's save the results as mean and std for each model (on the row) and source-target pair (on the column) as a latex table

example "
\begin{table*}[t]
\centering
\caption{Classification accuracy (\%) on Airport, Blog, and ArnetMiner datasets. The best results are highlighted in \textbf{bold} and the second best are \underline{underlined}.}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Models}} & \multicolumn{2}{c}{\textbf{Airport}} & \multicolumn{2}{c}{\textbf{Blog}} & \multicolumn{2}{c}{\textbf{ArnetMiner}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & E $\to$ U & U $\to$ E & B1 $\to$ B2 & B2 $\to$ B1 & C $\to$ A & D $\to$ A \\
\midrule
JHGDA & 36.89 $\pm$ 0.25 & 40.85 $\pm$ 1.68 & 17.79 $\pm$ 2.12 & 23.16 $\pm$ 6.59 & 65.53 $\pm$ 0.94 & 60.80 $\pm$ 0.35 \\
PairAlign & 42.38 $\pm$ 0.77 & 36.84 $\pm$ 1.48 & 32.17 $\pm$ 10.88 & 41.16 $\pm$ 3.02 & 58.06 $\pm$ 2.62 & 56.68 $\pm$ 0.89 \\
GRADE & 49.36 $\pm$ 0.35 & 48.45 $\pm$ 1.56 & \underline{38.64} $\pm$ 3.73 & \textbf{44.01} $\pm$ 4.51 & 69.16 $\pm$ 0.39 & 63.47 $\pm$ 1.10 \\
SpecReg & 37.59 $\pm$ 2.55 & 28.91 $\pm$ 8.77 & 28.27 $\pm$ 4.22 & 30.30 $\pm$ 1.35 & 68.90 $\pm$ 4.78 & 66.30 $\pm$ 4.28 \\
A2GNN & \underline{50.64} $\pm$ 1.47 & \underline{53.47} $\pm$ 0.24 & 22.58 $\pm$ 0.01 & 33.04 $\pm$ 4.12 & \textbf{76.15} $\pm$ 0.06 & \textbf{74.12} $\pm$ 0.18 \\
StructAlign & \textbf{54.34} $\pm$ 0.42 & \textbf{53.86} $\pm$ 1.21 & \textbf{52.93} $\pm$ 0.71 & \underline{42.98} $\pm$ 0.71 & \underline{70.68} $\pm$ 0.14 & \underline{66.92} $\pm$ 0.24 \\
\bottomrule
\end{tabular}
\end{table*}
"

"""

import os
import pandas as pd


def load_result(result_dir: str) -> pd.DataFrame:
    """result_dir: directory of one of the csv files"""

    file = pd.read_csv(result_dir)
    

    return file

def combine_results(result_dirs: list) -> pd.DataFrame:
    """result_dirs: list of directories of different csv files"""

    combined_df = pd.DataFrame()

    for result_dir in result_dirs:
        df = load_result(result_dir)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def save_latex_table(combined_df: pd.DataFrame, metric: str, save_path: str):

    combined_df['source_target'] = combined_df['source'].str[0] + combined_df['source'].str[-1] + '->' + combined_df['target'].str[0] + combined_df['target'].str[-1]


    mean_df = combined_df.groupby(['model', 'source_target'])[metric].mean().unstack()
    std_df = combined_df.groupby(['model', 'source_target'])[metric].std().unstack()
    latex_table = mean_df.copy()
    for col in latex_table.columns:
        latex_table[col] = mean_df[col].round(2).astype(str) + ' $\\pm$ ' + std_df[col].round(2).astype(str)
    latex_str = latex_table.to_latex(escape=False)
    with open(save_path, 'w') as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {save_path}")
    return combined_df

if __name__ == "__main__":
    result_dirs = [
        "./results/benchmark_0601_123456.csv",
        "./results/benchmark_0602_123456.csv",
        "./results/benchmark_0603_123456.csv",
    ]

    combined_df = combine_results(result_dirs)
    save_latex_table(combined_df, metric='accuracy', save_path='./results/benchmark_results.tex')  