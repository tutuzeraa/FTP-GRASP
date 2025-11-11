# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def parse_tsplib_euc2d(path: str) -> Dict:
    """Parse minimal TSPLIB EUC_2D to get name and coords (1-based ids)."""
    name = None
    coords: Dict[int, Tuple[float, float]] = {}
    in_coords = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NAME:"):
                name = line.split(":", 1)[1].strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if line.startswith("EOF"):
                break
            if in_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        i = int(float(parts[0]))
                        x = float(parts[1]); y = float(parts[2])
                        coords[i] = (x, y)
                    except Exception:
                        pass
    if not coords:
        raise ValueError("No coordinates parsed from TSPLIB file.")
    if name is None:
        name = Path(path).stem
    ids = sorted(coords.keys())
    xy = np.array([coords[i] for i in ids], dtype=float)
    return {"name": name, "ids": ids, "xy": xy}

def load_solution_csv(path: str) -> pd.DataFrame:
    """Load solution CSV with columns: node,parent,time (1-based)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for r in ["node", "parent", "time"]:
        if r not in cols:
            raise ValueError(f"CSV missing column '{r}' (found: {list(df.columns)})")
    df = df.rename(columns={cols["node"]: "node",
                            cols["parent"]: "parent",
                            cols["time"]: "time"})
    df["node"] = df["node"].astype(int)
    df["parent"] = df["parent"].astype(int)
    df["time"] = df["time"].astype(float)
    return df

def build_activation_tables(xy: np.ndarray, ids: List[int], df: pd.DataFrame):
    """
    Returns:
      times: array of activation times per node index (0-based aligned with ids order)
      parents: array of parents per node index (0-based index; -1 for root)
      root_idx: index of root (parent == -1)
    """
    n = len(ids)
    id2idx = {nid: i for i, nid in enumerate(ids)}
    times = np.full(n, np.inf, dtype=float)
    parents = np.full(n, -1, dtype=int)

    for _, row in df.iterrows():
        v_id = int(row["node"])
        p_id = int(row["parent"])
        t = float(row["time"])
        v = id2idx[v_id]
        times[v] = t
        parents[v] = id2idx[p_id] if p_id > 0 else -1

    roots = np.where(parents == -1)[0]
    root_idx = int(roots[0]) if len(roots) else int(np.argmin(times))
    return times, parents, root_idx

def animate_activation(tsp_path: str, csv_path: str, out_path: str,
                       fps: int = 24, seconds: int = 8, dpi: int = 140):
    """Create GIF animation showing activations over time."""
    inst = parse_tsplib_euc2d(tsp_path)
    df = load_solution_csv(csv_path)

    xy = inst["xy"]; ids = inst["ids"]; name = inst["name"]
    times, parents, root = build_activation_tables(xy, ids, df)

    T_max = float(np.nanmax(times[np.isfinite(times)]))
    if not math.isfinite(T_max) or T_max <= 0:
        T_max = float(np.nanmax(times[np.isfinite(times)] + 1e-9))
    total_frames = max(2, int(fps * seconds))

    # parent->child edges with (t_parent, t_child)
    edges = [(parents[v], v, times[parents[v]], times[v])
             for v in range(len(ids)) if parents[v] >= 0]
    edges.sort(key=lambda e: e[3])  # draw in arrival order

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    pad = 0.05
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_title(f"{name} — Activation over time")

    # inactive (faint) + active scatter (no explicit colors)
    scat_inactive = ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.15)
    scat_active   = ax.scatter([], [], s=50, alpha=1.0)

    # one line artist per edge; initially empty
    lines = [ax.plot([], [], linewidth=1.5, alpha=0.9)[0] for _ in edges]

    # id labels (small)
    _ = [ax.text(xy[i, 0], xy[i, 1], f"{nid}", fontsize=6,
                 ha="center", va="center", alpha=0.6)
         for i, nid in enumerate(ids)]

    time_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    active_mask = np.zeros(len(ids), dtype=bool)

    def frame_to_time(f: int) -> float:
        return (f / (total_frames - 1)) * T_max

    def update(frame: int):
        t_now = frame_to_time(frame)
        np.copyto(active_mask, times <= t_now)
        scat_active.set_offsets(xy[active_mask])

        # draw each edge partially from parent at t_p to child at t_v
        for k, (p, v, tp, tv) in enumerate(edges):
            x0, y0 = xy[p]; x1, y1 = xy[v]
            if t_now <= tp:
                lines[k].set_data([], [])
            elif tp < t_now < tv and tv > tp:
                s = (t_now - tp) / (tv - tp)
                xm = x0 + s * (x1 - x0); ym = y0 + s * (y1 - y0)
                lines[k].set_data([x0, xm], [y0, ym])
            else:
                if t_now >= tv:
                    lines[k].set_data([x0, x1], [y0, y1])
                else:
                    lines[k].set_data([], [])

        time_txt.set_text(f"t = {t_now:0.2f} / {T_max:0.2f}")
        return lines + [scat_active, time_txt]

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=True)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    PillowWriter(fps=fps).setup(fig, out_path.as_posix())  # ensure writer availability
    anim.save(out_path.as_posix(), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path.as_posix()

outdir = "animations/"
# animate_activation("instances_tsp/a280.tsp", "results/a280/a280.grasp.csv", outdir + "a280.gif")
animate_activation("instances_tsp/berlin52.tsp", "results/berlin52/berlin52.grasp.csv", outdir + "berlin52.gif")
# animate_activation("instances_analytic/star_n64.tsp", "results/star_n64/star_n64.grasp.csv", outdir + "star_n64.gif")
# animate_activation("instances_analytic/hypercube_embed_d5_n32.tsp", "results/hypercube_embed_d5_n32/hypercube_embed_d5_n32.grasp.csv", outdir + "hypercube_embed_d5_n32.gif")
# animate_activation("instances_analytic/debruijn_embed_k7_m2_n128.tsp", "results/debruijn_embed_k7_m2_n128/debruijn_embed_k7_m2_n128.grasp.csv", outdir + "debruijn_embed_k7_m2_n128.gif")
