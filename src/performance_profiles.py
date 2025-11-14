#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_RESULTS_DIR = "../results"

# Map from suffix in "Configuracao" to a nice strategy name
STRATEGY_MAP = {
    "_reactive_a0.00": "Reactive (α=0.00)",
    "_base_a0.05": "Base (α=0.05)",
    "_base_a0.10": "Base (α=0.10)",
    "_a0.05_2opt_pr": "2-opt + PR (α=0.05)",
    "_a0.05_2opt_pr_centr0.1": "2-opt + PR + λc=0.1",
    "_a0.05_2opt_pr_cong0.2": "2-opt + PR + λk=0.2",
    "_a0.05_2opt_pr_centr0.2_cong0.5": "2-opt + PR + λc=0.2, λk=0.5",
}

EXPECTED_CONFIGS = list(STRATEGY_MAP.keys())


def collect_data(base_results_dir: str) -> pd.DataFrame:
    """
    Reads every results/<instance>/_summary_report.csv and returns a long
    dataframe with columns: Instance, Strategy, Value (makespan).
    """
    base = Path(base_results_dir)
    if not base.is_dir():
        raise RuntimeError(f"Results directory '{base_results_dir}' not found.")

    rows = []

    for inst_dir in sorted(base.iterdir()):
        if not inst_dir.is_dir():
            continue

        summary_path = inst_dir / "_summary_report.csv"
        if not summary_path.is_file():
            print(f"[WARN] No _summary_report.csv in {inst_dir}, skipping.")
            continue

        instance_name = inst_dir.name

        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            # We'll put the rows in a dict first for easier lookup by Configuracao
            per_config = {}
            for row in reader:
                cfg = row.get("Configuracao")
                if cfg is None:
                    continue
                per_config[cfg] = row

        for cfg_suffix, label in STRATEGY_MAP.items():
            row = per_config.get(cfg_suffix)
            if row is None:
                value = np.nan
            else:
                try:
                    value = float(row.get("Makespan"))
                except (TypeError, ValueError):
                    value = np.nan

            rows.append(
                {
                    "Instance": instance_name,
                    "Strategy": label,
                    "Value": value,
                }
            )

    if not rows:
        raise RuntimeError("No data collected from results directory.")

    return pd.DataFrame(rows)


def build_ratios(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    From a long dataframe (Instance, Strategy, Value),
    build a wide matrix and compute performance ratios.

    Minimization:
        best_p = min_s Value_{p,s}
        r_{p,s} = Value_{p,s} / best_p
    """
    # Rows = instances, columns = strategies, values = makespan
    all_data = all_df.pivot(index="Instance", columns="Strategy", values="Value")

    # Best (minimum) per instance over strategies, skipping NaN
    best_values = all_data.min(axis=1, skipna=True)

    # Compute r_{p,s}
    ratios = all_data.values / best_values.values[:, np.newaxis]
    ratios_df = pd.DataFrame(ratios, index=all_data.index, columns=all_data.columns)

    return ratios_df


def safe_filename_from_label(label: str) -> str:
    """
    Turn a pretty label into something filename-safe and short-ish.
    """
    s = label.lower()
    replacements = {
        "α": "a",
        "λ": "lambda",
        "(": "",
        ")": "",
        ",": "",
        "+": "plus",
        "=": "",
        "  ": " ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = s.replace(" ", "_")
    s = s.replace(".", "")
    return s


def plot_per_strategy(ratios_df: pd.DataFrame, out_dir: Path):
    """
    For each strategy (column in ratios_df), generate a performance profile
    over all instances: one PNG per strategy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for strategy in ratios_df.columns:
        solver_ratios = ratios_df[strategy].dropna()

        if solver_ratios.empty:
            print(f"[WARN] No data for strategy '{strategy}', skipping.")
            continue

        # All finite ratios for THIS strategy
        finite = solver_ratios[np.isfinite(solver_ratios)]
        finite = finite[finite > 0]
        if finite.empty:
            print(f"[WARN] No finite ratios for strategy '{strategy}', skipping.")
            continue

        plot_taus = np.unique(finite[finite >= 1.0])
        if 1.0 not in plot_taus:
            plot_taus = np.insert(plot_taus, 0, 1.0)

        num_instances = len(solver_ratios)

        y_values = []
        for tau in plot_taus:
            count = (solver_ratios <= tau).sum()
            y_values.append(count / num_instances)

        plt.figure(figsize=(10, 6))
        plt.plot(
            plot_taus,
            y_values,
            drawstyle="steps-post",
            linewidth=2,
            label=strategy,
        )

        plt.title(f"Performance Profile - {strategy}")
        plt.xlabel("Performance Factor (τ)")
        plt.ylabel("Proportion of Problems  P(rₚ,s ≤ τ)")

        plt.xlim(left=1.0)
        plt.ylim(0, 1.05)

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()

        safe_name = safe_filename_from_label(strategy)
        out_path = out_dir / f"perf_profile_{safe_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Saved performance profile for '{strategy}' to: {out_path}")


def main():
    all_df = collect_data(BASE_RESULTS_DIR)
    ratios_df = build_ratios(all_df)
    out_dir = Path(BASE_RESULTS_DIR) / "per_strategy"
    plot_per_strategy(ratios_df, out_dir)


if __name__ == "__main__":
    main()
