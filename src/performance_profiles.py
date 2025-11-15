#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_RESULTS_DIR = "results"

# Mapear os sufixos do _summary_report para nomes bonitos
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
    Lê todos results/<instancia>/_summary_report.csv e devolve um DataFrame:
        Instance | Strategy | Value (makespan)
    """
    base = Path(base_results_dir)
    if not base.is_dir():
        raise RuntimeError(f"Diretório '{base_results_dir}' não encontrado.")

    rows = []

    for inst_dir in sorted(base.iterdir()):
        if not inst_dir.is_dir():
            continue

        summary_path = inst_dir / "_summary_report.csv"
        if not summary_path.is_file():
            print(f"[WARN] Sem _summary_report.csv em {inst_dir}, pulando.")
            continue

        instance_name = inst_dir.name

        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
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
        raise RuntimeError("Nenhum dado coletado em 'results/'.")

    return pd.DataFrame(rows)


def build_ratios(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir do DF longo (Instance, Strategy, Value),
    monta matriz instancia × estrategia e calcula os rácios.

    Minimização:
        best_p = min_s Value_{p,s}
        r_{p,s} = Value_{p,s} / best_p
    """
    # Linhas = instâncias, colunas = estratégias, valores = makespan
    all_data = all_df.pivot(index="Instance", columns="Strategy", values="Value")

    # Melhor valor por instância (mínimo entre estratégias)
    best_values = all_data.min(axis=1, skipna=True)

    # r_{p,s} = value / best
    ratios = all_data.values / best_values.values[:, np.newaxis]
    ratios_df = pd.DataFrame(ratios, index=all_data.index, columns=all_data.columns)

    return ratios_df


def plot_all_strategies(ratios_df: pd.DataFrame, output_path: Path):
    """
    Gera UM gráfico de performance profile com TODAS as estratégias.
    Cada linha é uma estratégia, eixo x = τ, eixo y = P(r_{p,s} <= τ).
    """
    # Todos os rácios finitos > 0 (todas estratégias, todas instâncias)
    finite = ratios_df.to_numpy().flatten()
    finite = finite[np.isfinite(finite)]
    finite = finite[finite > 0]

    if finite.size == 0:
        raise RuntimeError("Nenhum rácio finito para plotar.")

    plot_taus = np.unique(finite[finite >= 1.0])
    if 1.0 not in plot_taus:
        plot_taus = np.insert(plot_taus, 0, 1.0)

    plt.figure(figsize=(10, 6))

    # Para cada estratégia, calcula P(r_{p,s} <= τ)
    for strategy in ratios_df.columns:
        solver_ratios = ratios_df[strategy].dropna()
        if solver_ratios.empty:
            print(f"[WARN] Sem dados para estratégia '{strategy}', pulando.")
            continue

        num_instances = len(solver_ratios)
        y_values = []

        for tau in plot_taus:
            count = (solver_ratios <= tau).sum()
            y_values.append(count / num_instances)

        plt.plot(
            plot_taus,
            y_values,
            drawstyle="steps-post",
            linewidth=2,
            label=strategy,
        )

    plt.title("Performance Profile (Todas as Estratégias, Métrica: Makespan)")
    plt.xlabel("Fator de Desempenho (τ)")
    plt.ylabel("Proporção de Problemas  P(rₚ,s ≤ τ)")

    plt.xlim(left=1.0)
    plt.ylim(0, 1.05)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[OK] Gráfico combinado salvo em: {output_path}")


def main():
    all_df = collect_data(BASE_RESULTS_DIR)
    ratios_df = build_ratios(all_df)
    out_path = Path(BASE_RESULTS_DIR) / "performance_profile_all_strategies.png"
    plot_all_strategies(ratios_df, out_path)


if __name__ == "__main__":
    main()
