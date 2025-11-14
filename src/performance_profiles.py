#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_RESULTS_DIR = "../results"

# Map from suffix in "Configuracao" to pretty strategy names
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


def read_summary_file(summary_path: Path):
    """
    Reads _summary_report.csv and returns a dict:
        {config_suffix: makespan}
    Only keeps rows whose 'Configuracao' is in EXPECTED_CONFIGS.
    """
    makespans = {}
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row.get("Configuracao")
            if config not in EXPECTED_CONFIGS:
                continue
            try:
                mk = float(row.get("Makespan"))
            except (TypeError, ValueError):
                continue
            makespans[config] = mk
    return makespans


def plot_performance_profile_instance(instance_dir: Path):
    summary_path = instance_dir / "_summary_report.csv"
    if not summary_path.is_file():
        print(f"[WARN] No _summary_report.csv in {instance_dir}, skipping.")
        return

    instance_name = instance_dir.name
    print(f"[INFO] Building performance profile for instance: {instance_name}")

    makespans = read_summary_file(summary_path)
    if not makespans:
        print(f"[WARN] No known strategies found in {summary_path}, skipping.")
        return

    # For this single instance, best strategy is the one with MIN makespan
    best_makespan = min(makespans.values())

    # r_s = makespan_s / best_makespan
    ratios = {}
    for config, mk in makespans.items():
        ratios[config] = mk / best_makespan

    # Build τ values from ratios of this instance
    finite_ratios = np.array(list(ratios.values()), dtype=float)
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
    finite_ratios = finite_ratios[finite_ratios > 0]

    plot_taus = np.unique(finite_ratios[finite_ratios >= 1.0])
    if 1.0 not in plot_taus:
        plot_taus = np.insert(plot_taus, 0, 1.0)

    # With 1 instance: P(r_s <= τ) is 0 or 1
    num_instances = 1

    plt.figure(figsize=(10, 6))

    for config in EXPECTED_CONFIGS:
        if config not in ratios:
            continue  # this strategy missing for this instance

        ratio = ratios[config]
        y_values = []
        for tau in plot_taus:
            count = 1 if ratio <= tau else 0
            y_values.append(count / num_instances)  # i.e., 0 or 1

        label = STRATEGY_MAP.get(config, config)
        plt.plot(
            plot_taus,
            y_values,
            drawstyle="steps-post",
            linewidth=2,
            label=label,
        )

    plt.title(f"Performance Profile - {instance_name} (Makespan)")
    plt.xlabel("Performance Factor (τ)")
    plt.ylabel("Proportion of Problems  P(rₚ,s ≤ τ)")

    plt.xlim(left=1.0)
    plt.ylim(0, 1.05)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = instance_dir / f"performance_profile_{instance_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved: {out_path}")


def main():
    base = Path(BASE_RESULTS_DIR)
    if not base.is_dir():
        print(f"[ERROR] Results directory '{BASE_RESULTS_DIR}' not found.")
        return

    for child in sorted(base.iterdir()):
        if child.is_dir():
            plot_performance_profile_instance(child)


if __name__ == "__main__":
    main()
