#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Sequence
import csv
import os

# ============================================================
# Instance parsing (TSPLIB EUC_2D or simple CSV of x,y)
# ============================================================

@dataclass
class FTPInstance:
    name: str
    coords: List[Tuple[float, float]]  # nodes 0..n-1

    def n(self) -> int:
        return len(self.coords)

    def dist(self, i: int, j: int) -> float:
        (xi, yi) = self.coords[i]
        (xj, yj) = self.coords[j]
        return math.hypot(xi - xj, yi - yj)  # Euclidean (speed 1)

def parse_tsplib_euc2d(path: str) -> FTPInstance:
    name = os.path.basename(path)
    in_coords = False
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            U = line.upper()
            if not in_coords:
                if U.startswith("NAME"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                elif U.startswith("EDGE_WEIGHT_TYPE"):
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[1].strip().upper() != "EUC_2D":
                        raise ValueError("Only EUC_2D is supported")
                elif U.startswith("NODE_COORD_SECTION"):
                    in_coords = True
                elif U.startswith("EOF"):
                    break
                continue
            if in_coords:
                if U.startswith("EOF"):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        _ = int(parts[0])  # id; ignore order value
                        x = float(parts[1]); y = float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        pass
    if not coords:
        raise ValueError("No coordinates parsed.")
    return FTPInstance(name=name, coords=coords)

def parse_xy_csv(path: str) -> FTPInstance:
    # CSV: lines "x,y"
    coords = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            x,y = map(float, line.split(","))
            coords.append((x,y))
    return FTPInstance(name=os.path.basename(path), coords=coords)

# ============================================================
# Stop criteria
# ============================================================

class StopCriterion:
    def start(self): pass
    def update(self, iteration: int, best_cost: float, improved: bool): pass
    def should_stop(self) -> bool: return False

class MaxIterations(StopCriterion):
    def __init__(self, max_iters: int): self.max_iters=max_iters; self.iteration=0
    def update(self, iteration: int, best_cost: float, improved: bool): self.iteration=iteration
    def should_stop(self) -> bool: return self.iteration >= self.max_iters

class MaxTime(StopCriterion):
    def __init__(self, max_seconds: float): self.max_seconds=max_seconds; self.t0=None
    def start(self): self.t0=time.time()
    def should_stop(self) -> bool: return self.t0 is not None and (time.time()-self.t0)>=self.max_seconds

class NoImprovementIterations(StopCriterion):
    def __init__(self, max_no_improv: int): self.max_no_improv=max_no_improv; self.count=0
    def update(self, iteration: int, best_cost: float, improved: bool): self.count=0 if improved else self.count+1
    def should_stop(self) -> bool: return self.count >= self.max_no_improv

class CompositeStop(StopCriterion):
    def __init__(self, criteria: Iterable[StopCriterion]): self.criteria=list(criteria)
    def start(self): [c.start() for c in self.criteria]
    def update(self, iteration: int, best_cost: float, improved: bool):
        for c in self.criteria: c.update(iteration, best_cost, improved)
    def should_stop(self) -> bool: return any(c.should_stop() for c in self.criteria)

# ============================================================
# FTP solution representation and evaluation
# ============================================================

@dataclass
class FTPSolution:
    parent: List[int]        # parent[v] = parent of v; parent[root] = -1
    t: List[float]           # activation time of each node
    root: int
    makespan: float

def recompute_times(inst: FTPInstance, parent: Sequence[int], root: int) -> FTPSolution:
    """Compute activation times assuming each node departs immediately upon activation.
       Time(v) = Time(parent[v]) + dist(parent[v], v)."""
    n = inst.n()
    order = topo_order(parent, root)
    t = [float("inf")]*n
    t[root] = 0.0
    for v in order:
        if v == root: continue
        p = parent[v]
        t[v] = t[p] + inst.dist(p, v)
    return FTPSolution(parent=list(parent), t=t, root=root, makespan=max(t))

def topo_order(parent: Sequence[int], root: int) -> List[int]:
    n = len(parent)
    children = [[] for _ in range(n)]
    for v in range(n):
        p = parent[v]
        if v!=root and p!=-1: children[p].append(v)
    order=[]; stack=[root]
    while stack:
        u=stack.pop()
        order.append(u)
        stack.extend(children[u])
    # ensure parents appear before children; root first
    return order

# Subtree nodes of 'v' (including v)
def subtree_nodes(parent: Sequence[int], v: int) -> List[int]:
    n=len(parent)
    children=[[] for _ in range(n)]
    root = -1
    for u in range(n):
        p=parent[u]
        if p==-1: root=u
        elif p!=-1: children[p].append(u)
    out=[]; st=[v]
    while st:
        x=st.pop(); out.append(x); st.extend(children[x])
    return out

# Recompute times incrementally after changing parent of 'v'
def reparent_and_eval(inst: FTPInstance, sol: FTPSolution, v: int, new_parent: int) -> float:
    n=inst.n()
    parent=sol.parent[:]
    # prevent cycles: cannot attach ancestor under its descendant
    if new_parent in subtree_nodes(parent, v):
        return float("inf")
    parent[v]=new_parent
    # incrementally recompute t for subtree of v
    t=sol.t[:]
    # recompute t[v] and propagate to its subtree
    from collections import deque
    queue=deque([v])
    while queue:
        u=queue.popleft()
        if u==sol.root:
            t[u]=0.0
        else:
            p=parent[u]
            t[u]=t[p] + inst.dist(p,u)
        for w in range(n):
            if parent[w]==u:
                queue.append(w)
    return max(t)

# ============================================================
# Centrality & congestion (simple, cheap signals)
# ============================================================

def radial_centrality(inst: FTPInstance, k: int = 10) -> List[float]:
    """Cheap proxy: average distance to k nearest neighbors (smaller -> more central).
       Normalize to [0,1] and invert so 'higher is better central'."""
    n=inst.n()
    k=min(k, max(1,n-1))
    avgs=[]
    for i in range(n):
        ds=[inst.dist(i,j) for j in range(n) if j!=i]
        ds.sort()
        avg=sum(ds[:k])/k
        avgs.append(avg)
    mx=max(avgs); mn=min(avgs)
    if mx==mn: return [0.5]*n
    # invert: small avg -> high centrality
    return [(mx - a)/(mx - mn) for a in avgs]

def current_congestion(parent: Sequence[int]) -> List[int]:
    n=len(parent)
    deg=[0]*n
    for v in range(n):
        p=parent[v]
        if p!=-1: deg[p]+=1
    return deg  # how many children a node currently serves

# ============================================================
# GRASP
# ============================================================

@dataclass
class GraspConfig:
    alpha: float = 0.2
    seed: Optional[int] = None
    reactive: bool = False
    reactive_grid: Tuple[float,...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    reactive_tau: int = 25  # update window

def greedy_randomized_construction(inst: FTPInstance, root: int, alpha: float,
                                   rng: random.Random,
                                   lambda_central: float,
                                   lambda_congest: float,
                                   centrality: Optional[Sequence[float]] = None) -> FTPSolution:
    n=inst.n()
    parent=[-1]*n
    t=[float("inf")]*n
    active= {root}
    t[root]=0.0

    # helper: earliest activation candidate for a node v
    def best_parent_and_time(v:int)->Tuple[int,float]:
        best_p=None; best= float("inf")
        for u in active:
            cand = t[u]+inst.dist(u,v)
            if cand<best: best=cand; best_p=u
        return best_p,best

    # construction loop
    while len(active)<n:
        # evaluate all inactive nodes
        cand_list=[]
        cong = current_congestion(parent)  # counts current children per active node
        for v in range(n):
            if v in active: continue
            p, tv = best_parent_and_time(v)
            # score pieces
            s_time = tv
            s_central = 0.0 if centrality is None else -centrality[v]  # higher centrality -> smaller score
            s_cong = cong[p] if p is not None else 0
            score = s_time + lambda_central*s_central + lambda_congest*s_cong
            cand_list.append((v, p, tv, score))
        cand_list.sort(key=lambda x: x[3])
        # RCL by alpha threshold
        smin=cand_list[0][3]; smax=cand_list[-1][3]
        thresh = smin + alpha*(smax - smin) if smax>smin else smin
        rcl=[c for c in cand_list if c[3] <= thresh]
        v,p,tv,_ = rng.choice(rcl)
        # commit
        parent[v]=p
        t[v]=tv
        active.add(v)
    return FTPSolution(parent=parent, t=t, root=root, makespan=max(t))

def local_search_reparent(inst: FTPInstance, sol: FTPSolution) -> FTPSolution:
    """First-improvement: try reparenting a node to any earlier-activated node."""
    rng = random.Random(0)
    n=inst.n()
    order = sorted(range(n), key=lambda v: sol.t[v])  # try earlier nodes first as parents
    improved=True
    parent=sol.parent[:]
    t=sol.t[:]
    best_ms=sol.makespan
    while improved:
        improved=False
        for v in range(n):
            if v==sol.root: continue
            best_local_ms = best_ms
            best_new_p = None
            for u in order:
                if u==v: continue
                if sol.t[u] > t[v]:  # only consider parents that are no later than current activation
                    break
                ms = reparent_and_eval(inst, FTPSolution(parent,t,sol.root,best_ms), v, u)
                if ms + 1e-9 < best_local_ms:
                    best_local_ms = ms
                    best_new_p = u
            if best_new_p is not None:
                # accept move
                parent[v]=best_new_p
                # recompute full times (cheap enough)
                new_sol = recompute_times(inst, parent, sol.root)
                parent = new_sol.parent[:]
                t = new_sol.t[:]
                best_ms = new_sol.makespan
                improved=True
                break
    return FTPSolution(parent=parent, t=t, root=sol.root, makespan=best_ms)

def grasp(inst: FTPInstance, root: int, cfg: GraspConfig, stopper: StopCriterion,
          lambda_central: float, lambda_congest: float, k_central: int,
          log_best: bool=False) -> Tuple[FTPSolution,int]:
    rng = random.Random(cfg.seed)
    n=inst.n()
    centrality = radial_centrality(inst, k=k_central) if lambda_central!=0.0 else None

    best: Optional[FTPSolution]=None
    iteration=0

    # reactive α
    grid = list(cfg.reactive_grid)
    probs = [1/len(grid)]*len(grid)
    perf_window=[]  # (alpha_idx, best_ms)

    stopper.start()
    while True:
        iteration += 1
        alpha = cfg.alpha
        if cfg.reactive:
            # sample alpha from current probs
            r = rng.random()
            cum=0.0
            choice=0
            for i,p in enumerate(probs):
                cum+=p
                if r<=cum: choice=i; break
            alpha = grid[choice]
        # construction + LS
        cand = greedy_randomized_construction(inst, root, alpha, rng,
                                              lambda_central, lambda_congest, centrality)
        cand = local_search_reparent(inst, cand)

        improved=False
        if best is None or cand.makespan < best.makespan - 1e-9:
            best = cand
            improved=True

        stopper.update(iteration, best.makespan if best else float("inf"), improved)
        if cfg.reactive:
            perf_window.append((alpha, cand.makespan))
            if len(perf_window) >= cfg.reactive_tau:
                # compute average perf per alpha; better (smaller) gets higher prob
                sums={a:[] for a in grid}
                for a,ms in perf_window: sums[a].append(ms)
                avgs=[(sum(sums[a])/len(sums[a])) if sums[a] else float("inf") for a in grid]
                # invert to scores
                mx = max(avgs)
                scores=[(mx - v + 1e-9) for v in avgs]  # avoid zeros
                total=sum(scores)
                probs=[s/total for s in scores]
                perf_window.clear()

        if stopper.should_stop():
            break

    return best, iteration

# ============================================================
# CLI
# ============================================================

def build_stopper(args: argparse.Namespace) -> StopCriterion:
    cs=[]
    if args.max_iters is not None: cs.append(MaxIterations(args.max_iters))
    if args.max_time is not None: cs.append(MaxTime(args.max_time))
    if args.max_no_improv is not None: cs.append(NoImprovementIterations(args.max_no_improv))
    if not cs: cs=[MaxIterations(1000)]
    return CompositeStop(cs)

def main():
    p=argparse.ArgumentParser(description="GRASP for Freeze-Tag Problem (geometric/EUC_2D)")
    p.add_argument("instance", help=".tsp (EUC_2D) or CSV (x,y)")
    p.add_argument("--root", type=int, default=1, help="1-based root index (default: 1)")
    p.add_argument("--format", choices=["auto","tsp","csv"], default="auto")
# GRASP
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reactive", action="store_true", default=False)
    # signals
    p.add_argument("--lambda-central", type=float, default=0.0, help="weight for centrality (higher -> favor central nodes)")
    p.add_argument("--k-central", type=int, default=10, help="#neighbors for radial centrality")
    p.add_argument("--lambda-congest", type=float, default=0.0, help="weight for congestion (higher -> penalize busy parents)")
    # stop criteria
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--max-no-improv", type=int, default=None)
    # outputs
    p.add_argument("--save-tree", action="store_true")
    p.add_argument("--save-csv", action="store_true")

    args=p.parse_args()
    fmt=args.format
    if fmt=="auto":
        fmt="tsp" if args.instance.lower().endswith(".tsp") else "csv"
    inst = parse_tsplib_euc2d(args.instance) if fmt=="tsp" else parse_xy_csv(args.instance)
    root = args.root-1
    if not (0<=root<inst.n()): raise ValueError("invalid root index")

    cfg = GraspConfig(alpha=args.alpha, seed=args.seed, reactive=args.reactive)
    stopper = build_stopper(args)

    t0=time.time()
    best, iters = grasp(inst, root, cfg, stopper,
                        lambda_central=args.lambda_central,
                        lambda_congest=args.lambda_congest,
                        k_central=args.k_central)
    elapsed=time.time()-t0

    # report
    print(f"Instance: {inst.name}")
    print(f"Nodes:    {inst.n()}   Root: {args.root}")
    print(f"Iters:    {iters}   Alpha: {cfg.alpha}   Reactive: {cfg.reactive}")
    print(f"Best ms:  {best.makespan:.6f}   Time(s): {elapsed:.3f}")

    # optional outputs
    if args.save_tree:
        out = args.instance + ".grasp.tree"
        with open(out,"w",encoding="utf-8") as w:
            w.write(f"NAME: {inst.name}\nTYPE: TREE\nROOT: {args.root}\n")
            w.write("PARENT_SECTION (1-based: node parent)\n")
            for v,pv in enumerate(best.parent):
                pb = 0 if pv==-1 else (pv+1)
                w.write(f"{v+1} {pb}\n")
        print(f"Saved tree: {out}")

    if args.save_csv:
        out = args.instance + ".grasp.csv"
        with open(out,"w",newline="",encoding="utf-8") as w:
            cw=csv.writer(w)
            cw.writerow(["node","parent","time"])
            for v in range(inst.n()):
                pv = -1 if best.parent[v]==-1 else best.parent[v]+1
                cw.writerow([v+1, pv, f"{best.t[v]:.6f}"])
        print(f"Saved CSV:  {out}")

if __name__=="__main__":
    main()
