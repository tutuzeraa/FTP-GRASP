#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# GRASP para o Freeze-Tag Problem (FTP) — implementação comentada
# ============================================================================
# Este arquivo implementa um GRASP multi-start alinhado à proposta em LaTeX:
# - Pré-processamento leve com distância euclidiana (EUC_2D) e um sinal de
#   centralidade radial (k-vizinhos mais próximos).
# - Construção gulosa-aleatorizada com RCL controlada por α, combinando:
#     score = tempo de ativação + λ_congest * congestionamento - λ_central * centralidade
# - Busca local por reanexação de pai (1-move), com recomputação incremental
#   dos tempos e prevenção de ciclos (evita anexar ancestral em descendente).
# - GRASP reativo (opcional), atualizando a distribuição sobre α.
# - Critérios de parada combináveis: iterações, tempo, e janela sem melhoria.
#
# Observação: evitamos variáveis não utilizadas; o código está limpo e coerente.
# ============================================================================

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Sequence
import csv
import os

# ============================================================
# Representação de instância e parsing
# ============================================================

@dataclass
class FTPInstance:
    # Estrutura de uma instância geométrica do FTP (EUC_2D).
    # coords: lista de pares (x, y) para nós 0..n-1.
    name: str
    coords: List[Tuple[float, float]]

    def n(self) -> int:
        # Retorna o número de vértices.
        return len(self.coords)

    def dist(self, i: int, j: int) -> float:
        # Distância euclidiana entre i e j. Velocidade unitária => tempo = distância.
        (xi, yi) = self.coords[i]
        (xj, yj) = self.coords[j]
        return math.hypot(xi - xj, yi - yj)

def parse_tsplib_euc2d(path: str) -> FTPInstance:
    # Parser simples para arquivos TSPLIB com EDGE_WEIGHT_TYPE = EUC_2D.
    # Lê o bloco NODE_COORD_SECTION e extrai (x,y).
    name = os.path.basename(path)
    in_coords = False
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            U = line.upper()
            # Trata cabeçalho até entrar na seção de coordenadas
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
            # Dentro da seção de coordenadas
            if in_coords:
                if U.startswith("EOF"):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        _ = int(parts[0])  # índice do nó (ignora; usamos ordem)
                        x = float(parts[1]); y = float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        # Linha inválida — ignora
                        pass
    if not coords:
        raise ValueError("No coordinates parsed.")
    return FTPInstance(name=name, coords=coords)

def parse_xy_csv(path: str) -> FTPInstance:
    # Lê CSV no formato: uma linha por ponto, 'x,y'.
    coords = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = map(float, line.split(","))
            coords.append((x, y))
    return FTPInstance(name=os.path.basename(path), coords=coords)

# ============================================================
# Critérios de parada (combináveis)
# ============================================================

class StopCriterion:
    # Interface base: permite combinar múltiplos critérios.
    def start(self):
        pass
    def update(self, iteration: int, best_cost: float, improved: bool):
        pass
    def should_stop(self) -> bool:
        return False

class MaxIterations(StopCriterion):
    # Para quando atinge um número máximo de iterações do GRASP.
    def __init__(self, max_iters: int):
        self.max_iters = max_iters
        self.iteration = 0
    def update(self, iteration: int, best_cost: float, improved: bool):
        self.iteration = iteration
    def should_stop(self) -> bool:
        return self.iteration >= self.max_iters

class MaxTime(StopCriterion):
    # Para por tempo total de execução (em segundos).
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.t0 = None
    def start(self):
        self.t0 = time.time()
    def should_stop(self) -> bool:
        return self.t0 is not None and (time.time() - self.t0) >= self.max_seconds

class NoImprovementIterations(StopCriterion):
    # Para quando há janelas longas sem melhoria do melhor custo.
    def __init__(self, max_no_improv: int):
        self.max_no_improv = max_no_improv
        self.count = 0
    def update(self, iteration: int, best_cost: float, improved: bool):
        self.count = 0 if improved else self.count + 1
    def should_stop(self) -> bool:
        return self.count >= self.max_no_improv

class CompositeStop(StopCriterion):
    # Combina múltiplos critérios com OU lógico: para quando qualquer um ativa.
    def __init__(self, criteria: Iterable[StopCriterion]):
        self.criteria = list(criteria)
    def start(self):
        for c in self.criteria:
            c.start()
    def update(self, iteration: int, best_cost: float, improved: bool):
        for c in self.criteria:
            c.update(iteration, best_cost, improved)
    def should_stop(self) -> bool:
        return any(c.should_stop() for c in self.criteria)

# ============================================================
# Representação de solução e avaliação de tempos
# ============================================================

@dataclass
class FTPSolution:
    # Representa uma árvore de ativação:
    #   - parent[v] = pai de v (root tem -1)
    #   - t[v] = instante de ativação de v
    #   - makespan = max_v t[v]
    parent: List[int]
    t: List[float]
    root: int
    makespan: float

def topo_order(parent: Sequence[int], root: int) -> List[int]:
    # Ordem topológica da árvore (pais antes dos filhos).
    # Implementação por DFS explícito usando lista de filhos.
    n = len(parent)
    children = [[] for _ in range(n)]
    for v in range(n):
        p = parent[v]
        if v != root and p != -1:
            children[p].append(v)
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        # empilha filhos de u; ordem não importa para a correção
        stack.extend(children[u])
    return order

def recompute_times(inst: FTPInstance, parent: Sequence[int], root: int) -> FTPSolution:
    # Recalcula t[v] para todos os vértices assumindo partida imediata do pai:
    #   t[v] = t[parent[v]] + dist(parent[v], v), com t[root] = 0.
    # Retorna FTPSolution com makespan consistente.
    n = inst.n()
    order = topo_order(parent, root)
    t = [float("inf")] * n
    t[root] = 0.0
    for v in order:
        if v == root:
            continue
        p = parent[v]
        t[v] = t[p] + inst.dist(p, v)
    return FTPSolution(parent=list(parent), t=t, root=root, makespan=max(t))

def subtree_nodes(parent: Sequence[int], v: int) -> List[int]:
    # Retorna todos os nós da subárvore enraizada em v (inclui v).
    # Usado para bloquear ciclos em reanexação (não anexar ancestral sob descendente).
    n = len(parent)
    children = [[] for _ in range(n)]
    for u in range(n):
        p = parent[u]
        if p != -1:
            children[p].append(u)
    out = []
    st = [v]
    while st:
        x = st.pop()
        out.append(x)
        st.extend(children[x])
    return out

def reparent_and_eval(inst: FTPInstance, sol: FTPSolution, v: int, new_parent: int) -> float:
    # Avalia mover o pai de v para new_parent.
    # - Impede ciclos: verifica se new_parent está dentro da subárvore de v.
    # - Recalcula t apenas na subárvore afetada (BFS) e retorna o makespan resultante.
    n = inst.n()
    parent = sol.parent[:]
    # Previne ciclo: não pode anexar ancestral abaixo do próprio descendente
    if new_parent in subtree_nodes(parent, v):
        return float("inf")
    parent[v] = new_parent

    # Recomputação incremental: propaga novos tempos a partir de v
    t = sol.t[:]
    from collections import deque
    queue = deque([v])
    while queue:
        u = queue.popleft()
        if u == sol.root:
            t[u] = 0.0
        else:
            p = parent[u]
            t[u] = t[p] + inst.dist(p, u)
        # Enfileira filhos de u no novo parent array
        for w in range(n):
            if parent[w] == u:
                queue.append(w)
    return max(t)

# ============================================================
# Restriçao de fan-out: utilidades
# ============================================================

def _init_capacity(n: int, root: int) -> list[int]:
    """Capacidade de ramificação por nó: raiz=1, demais=2."""
    cap = [2] * n
    cap[root] = 1
    return cap

def _compute_outdeg(parent: list[int]) -> list[int]:
    """Out-degree atual de cada nó, dado o vetor de pais (-1 na raiz)."""
    n = len(parent)
    out = [0] * n
    for v, p in enumerate(parent):
        if p >= 0:
            out[p] += 1
    return out

# ============================================================
# Sinais auxiliares: centralidade e congestionamento
# ============================================================

def radial_centrality(inst: FTPInstance, k: int = 10) -> List[float]:
    # Centralidade radial barata: média das distâncias aos k vizinhos mais próximos.
    # - Valores menores => posição mais central.
    # - Normalizamos em [0,1] e invertimos para "maior é melhor".
    n = inst.n()
    k = min(k, max(1, n - 1))
    avgs = []
    for i in range(n):
        ds = [inst.dist(i, j) for j in range(n) if j != i]
        ds.sort()
        avg = sum(ds[:k]) / k
        avgs.append(avg)
    mx = max(avgs); mn = min(avgs)
    if mx == mn:
        return [0.5] * n  # todos iguais
    return [(mx - a) / (mx - mn) for a in avgs]  # invertido: menor avg => maior valor

def current_congestion(parent: Sequence[int]) -> List[int]:
    # Congestionamento atual: número de filhos por nó (grau de saída na árvore).
    # Sinal usado para penalizar pais muito sobrecarregados na construção.
    n = len(parent)
    deg = [0] * n
    for v in range(n):
        p = parent[v]
        if p != -1:
            deg[p] += 1
    return deg

# ============================================================
# GRASP: configuração, construção e busca local
# ============================================================

@dataclass
class GraspConfig:
    # Parâmetros do GRASP:
    #   - alpha: tamanho efetivo da RCL (0 => guloso puro; 1 => aleatório uniforme).
    #   - seed: semente do gerador aleatório.
    #   - reactive: ativa GRASP reativo (α sorteado de uma grade com probabilidades aprendidas).
    #   - reactive_grid: valores possíveis de α no modo reativo.
    #   - reactive_tau: janela para atualizar probabilidades.
    alpha: float = 0.2
    seed: Optional[int] = None
    reactive: bool = False
    reactive_grid: Tuple[float,...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
    reactive_tau: int = 25

def greedy_randomized_construction(inst: FTPInstance, root: int, alpha: float,
                                   rng: random.Random,
                                   lambda_central: float,
                                   lambda_congest: float,
                                   centrality: Optional[Sequence[float]] = None) -> FTPSolution:
    # Fase de construção do GRASP:
    #   - Mantém conjunto de nós ativos.
    #   - Para cada inativo, estima melhor pai ativo e tempo de ativação (partida imediata).
    #   - Score(v) = tempo + λ_congest * congest(parent) - λ_central * centralidade[v].
    #   - Monta RCL por limiar (α) e escolhe aleatoriamente um candidato.
    n = inst.n()
    parent = [-1] * n
    t = [float("inf")] * n

    # estado inicial
    t[root] = 0.0
    active = set([root])              # nós já ativados
    capacity = _init_capacity(n, root)
    outdeg = [0] * n
    eligible = set([root])            # ativos com outdeg < capacity

    # sinais auxiliares (se usados)
    # TODO: Arrumar isso
    central = [0.0] * n
    if lambda_central != 0.0:
        central = radial_centrality(inst)

    # enquanto houver inativos
    while len(active) < n:
        if not eligible:
            raise RuntimeError("Sem pais elegíveis durante a construção (verifique capacidade).")

        candidates = []
        # para cada v inativo, escolha melhor pai ENTRE os elegíveis
        for v in range(n):
            if v in active:
                continue
            best_p = None
            best_arrival = None
            best_cong = 0

            # melhor pai = elegível que chega mais cedo
            for u in eligible:
                arr = t[u] + inst.dist(u, v)
                if (best_arrival is None) or (arr < best_arrival):
                    best_arrival = arr
                    best_p = u

            # score = chegada + λ_congest * outdeg(pai) - λ_central * centralidade[v]
            cong = outdeg[best_p]
            score = best_arrival + (lambda_congest * cong) - (lambda_central * central[v])
            candidates.append((score, v, best_p, best_arrival))

        # RCL por alpha (fração do topo)
        candidates.sort(key=lambda x: x[0])
        if alpha <= 0.0:
            chosen = candidates[0]
        else:
            k = max(1, int(math.ceil(alpha * len(candidates))))
            idx = int(rng.randrange(k))
            chosen = candidates[idx]

        _, v_star, u_star, arrival = chosen

        # anexa v_star ao pai u_star
        parent[v_star] = u_star
        t[v_star] = arrival
        active.add(v_star)

        # atualiza outdeg/capacidade/eligibilidade
        outdeg[u_star] += 1
        if outdeg[u_star] >= capacity[u_star] and u_star in eligible:
            eligible.remove(u_star)

        # novo nó ativado entra com capacidade 2 e outdeg 0 -> torna-se elegível
        # (v_star nunca é a raiz aqui)
        if outdeg[v_star] < capacity[v_star]:
            eligible.add(v_star)

    makespan = max(t)
    return FTPSolution(parent=parent, t=t, root=root, makespan=makespan)


def local_search_reparent(inst: FTPInstance, sol: FTPSolution, root:int) -> FTPSolution:
    # Busca local por reanexação de subárvore (1-move), estratégia de primeira melhora:
    #   - Tenta para cada v != root reanexar em pais elegíveis.
    #   - Avalia movimento com reparent_and_eval (incremental) [O(subtree_size]).
    #   - Ao aceitar, recomputa tempos globalmente (recompute_times) para estabilizar
    #     o estado 'sol' e recomeça a varredura.
    n = inst.n()
    # Copiamos a solução base para esta 'passada' da busca local
    current_sol = FTPSolution(parent=sol.parent[:], t=sol.t[:], root=sol.root, makespan=sol.makespan)
    
    capacity = _init_capacity(n, root)
    outdeg = _compute_outdeg(current_sol.parent)

    improved = True
    while improved:
        improved = False
        
        # O makespan a ser batido nesta iteração da busca
        best_T = current_sol.makespan

        # ordem: nós mais críticos primeiro (maior tempo)
        order = sorted(range(n), key=lambda v: current_sol.t[v], reverse=True)

        for v in order:
            if v == root:
                continue

            old_p = current_sol.parent[v]
            
            # Subárvore de v para evitar ciclos (não podemos anexar v em um de seus filhos)
            # Nota: reparent_and_eval JÁ faz essa checagem, mas é bom filtrar
            # os candidatos antes para economizar chamadas.
            sub = set(subtree_nodes(current_sol.parent, v))
            
            # candidatos: ativos, com capacidade sobrando, não na subárvore de v
            cand_parents = [u for u in range(n)
                            if (current_sol.t[u] < float("inf")) and \
                               (outdeg[u] < capacity[u]) and \
                               (u not in sub) and \
                               (u != v)]
            best_move = None

            for u in cand_parents:
                if u == old_p:
                    continue
                
                # =========================================================
                # MUDANÇA PRINCIPAL: AVALIAÇÃO INCREMENTAL
                # =========================================================
                # Em vez de recompute_times, usamos a avaliação incremental
                # que só recalcula a subárvore de 'v'.
                # O 'current_sol' (baseline) não é modificado aqui.
                trial_T = reparent_and_eval(inst, current_sol, v, u)
                
                if trial_T + 1e-12 < best_T:  # aceita melhora estrita (tolerância numérica)
                    best_T = trial_T
                    best_move = (v, old_p, u)
                    break  # first-improvement

            if best_move is not None:
                v_move, old_p_move, u_move = best_move
                
                # =========================================================
                # APLICA O MOVIMENTO E RE-ESTABILIZA
                # =========================================================
                # Agora que achamos uma melhora, aplicamos o 'parent'
                current_sol.parent[v_move] = u_move
                
                # E recalculamos a árvore *uma vez* para ter os t[v] corretos
                # para a próxima iteração do 'while improved'.
                current_sol = recompute_times(inst, current_sol.parent, root)
                
                # Atualiza outdeg (necessário recalcular pois 'sol' é novo)
                outdeg = _compute_outdeg(current_sol.parent)

                improved = True
                break  # recomeça varredura a partir da nova solução 'current_sol'

    # Retorna a solução localmente ótima encontrada
    return current_sol

# ============================================================
# Path Relinking
# ============================================================

def path_relinking(inst: FTPInstance, sol_start: FTPSolution, sol_guide: FTPSolution) -> FTPSolution:
    """
    Explora o caminho de 'sol_start' até 'sol_guide', aplicando um 1-move de 
    cada vez. Retorna a melhor solução (menor makespan) encontrada *no caminho*.
    
    sol_start: Solução de partida (ex: a candidata atual).
    sol_guide: Solução guia (ex: a melhor global).
    """
    # A melhor solução no caminho é, no mínimo, a de partida.
    best_on_path = sol_start
    
    # Começamos com uma cópia da solução inicial para modificar
    current_parent = sol_start.parent[:]
    
    # 1. Encontra o "delta": conjunto de nós com pais diferentes
    delta = set()
    n = inst.n()
    for v in range(n):
        if current_parent[v] != sol_guide.parent[v]:
            delta.add(v)
            
    # 2. Caminha enquanto houver diferenças
    while delta:
        # Pega um nó aleatório (ou o primeiro) para mover
        # .pop() é determinístico e eficiente.
        v_to_move = delta.pop()
        
        # O pai que este nó 'deveria' ter, segundo a solução guia
        target_parent = sol_guide.parent[v_to_move]
        
        # Aplica o movimento
        current_parent[v_to_move] = target_parent
        
        # 3. Avalia o resultado deste 1-move
        # Usamos recompute_times para garantir um estado 100% correto
        # (pais, tempos e makespan) para o próximo loop.
        intermediate_sol = recompute_times(inst, current_parent, sol_start.root)
        
        # 4. Verifica se esta solução no meio do caminho é uma nova
        #    melhor solução global.
        if intermediate_sol.makespan < best_on_path.makespan - 1e-9:
            best_on_path = intermediate_sol
            
    # Retorna a melhor solução encontrada durante a travessia
    return best_on_path

def grasp(inst: FTPInstance, root: int, cfg: GraspConfig, stopper: StopCriterion,
          lambda_central: float, lambda_congest: float, k_central: int,
          use_path_relinking: bool = False): # <-- NOVO ARGUMENTO
    # Loop principal do GRASP:
    #   - (Opcional) pré-computa centralidade radial.
    #   - Itera construção + busca local.
    #   - (OpcFional) Path-Relinking entre 'cand' e 'best'.
    #   - Mantém melhor solução global.
    #   - (Opcional) atualiza probabilidades de α (GRASP reativo) por janela.
    # Retorna (melhor_solucao, numero_de_iteracoes).
    rng = random.Random(cfg.seed)

    centrality = radial_centrality(inst, k=k_central) if lambda_central != 0.0 else None

    best: Optional[FTPSolution] = None
    iteration = 0

    # Dados do reativo
    grid = list(cfg.reactive_grid)
    probs = [1 / len(grid)] * len(grid)
    perf_window = []  # janela de (alpha, makespan)

    stopper.start()
    while True:
        iteration += 1
        # Escolha de α: fixo ou amostrado do reativo
        alpha = cfg.alpha
        if cfg.reactive:
            r = rng.random()
            cum = 0.0
            choice = 0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    choice = i
                    break
            alpha = grid[choice]

        # 1) Construção
        cand = greedy_randomized_construction(inst, root, alpha, rng,
                                              lambda_central, lambda_congest, centrality)
        # 2) Busca Local
        cand = local_search_reparent(inst, cand, root)

        # ============================================================
        # BLOCO NOVO: Path-Relinking (Intensificação)
        # ============================================================
        pr_cand = cand # A candidata padrão é o resultado da busca local
        
        # Se o PR estiver ativo E já tivermos uma 'best' para guiar
        if use_path_relinking and best is not None:
            # Tenta encontrar uma solução no caminho entre 'cand' e 'best'
            pr_cand = path_relinking(inst, cand, best)
        
        # A candidata final (pr_cand) agora é o melhor resultado
        # da (Busca Local) OU do (Path Relinking)
        # ============================================================

        improved = False
        # Comparamos a *melhor* candidata (pr_cand) com a global (best)
        if best is None or pr_cand.makespan < best.makespan - 1e-9:
            best = pr_cand # Salva a melhor candidata encontrada
            improved = True

        # Atualiza critérios de parada
        stopper.update(iteration, best.makespan if best else float("inf"), improved)

        # Atualização do reativo: melhora aumentam probabilidade dos α vencedores
        if cfg.reactive:
            perf_window.append((alpha, cand.makespan)) # Nota: usa o 'cand' original para o score de alpha
            if len(perf_window) >= cfg.reactive_tau:
                sums = {a: [] for a in grid}
                for a, ms in perf_window:
                    sums[a].append(ms)
                avgs = [(sum(sums[a]) / len(sums[a])) if sums[a] else float("inf") for a in grid]
                mx = max(avgs)
                # transforma custo em score (inversão simples) e normaliza
                scores = [(mx - v + 1e-9) for v in avgs]
                total = sum(scores)
                probs = [s / total for s in scores]
                perf_window.clear()

        if stopper.should_stop():
            break

    return best, iteration
    
# ============================================================
# CLI e integração com critérios de parada
# ============================================================

def build_stopper(args: argparse.Namespace) -> StopCriterion:
    # Constrói um CompositeStop a partir das flags da linha de comando.
    # Defaults seguros: MaxIterations(1000) se nada for definido.
    cs = []
    if args.max_iters is not None:
        cs.append(MaxIterations(args.max_iters))
    if args.max_time is not None:
        cs.append(MaxTime(args.max_time))
    if args.max_no_improv is not None:
        cs.append(NoImprovementIterations(args.max_no_improv))
    if not cs:
        cs = [MaxIterations(1000)]
    return CompositeStop(cs)

def main():
    # Ponto de entrada:
    #   - Lê instância (.tsp EUC_2D ou .csv x,y).
    #   - Executa GRASP com parâmetros fornecidos.
    #   - Reporta melhor makespan e persiste (opcional) árvore e CSV de tempos.
    p = argparse.ArgumentParser(description="GRASP for Freeze-Tag Problem (geometric/EUC_2D)")
    p.add_argument("instance", help=".tsp (EUC_2D) or CSV (x,y)")
    p.add_argument("--root", type=int, default=1, help="1-based root index (default: 1)")
    p.add_argument("--format", choices=["auto", "tsp", "csv"], default="auto")
    # GRASP
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reactive", action="store_true", default=False)
    p.add_argument("--path-relinking", action="store_true", default=False, help="Ativa path-relinking")
    # sinais
    p.add_argument("--lambda-central", type=float, default=0.0, help="peso da centralidade (maior => favorece centrais)")
    p.add_argument("--k-central", type=int, default=10, help="#vizinhos para centralidade radial")
    p.add_argument("--lambda-congest", type=float, default=0.0, help="peso do congestionamento (maior => penaliza pais ocupados)")
    # parada
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--max-no-improv", type=int, default=None)
    # saídas
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--save-tree", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--suffix", type=str, default="", help="Sufixo para os experimentos (ex: _exp1)")

    args = p.parse_args()
    fmt = args.format
    if fmt == "auto":
        fmt = "tsp" if args.instance.lower().endswith(".tsp") else "csv"
    inst = parse_tsplib_euc2d(args.instance) if fmt == "tsp" else parse_xy_csv(args.instance)
    root = args.root - 1  # converte para 0-based interno
    if not (0 <= root < inst.n()):
        raise ValueError("invalid root index")

    cfg = GraspConfig(alpha=args.alpha, seed=args.seed, reactive=args.reactive)
    stopper = build_stopper(args)

    t0 = time.time()
    best, iters = grasp(inst, root, cfg, stopper,
                        lambda_central=args.lambda_central,
                        lambda_congest=args.lambda_congest,
                        k_central=args.k_central,
                        use_path_relinking=args.path_relinking) # <-- PASSA O NOVO FLAG
    elapsed = time.time() - t0 

    out_path = args.outdir / inst.name
    out_path.mkdir(parents=True, exist_ok=True)

    report_lines = [
        f"Instance: {inst.name}",
        f"Nodes:    {inst.n()}   Root: {args.root}",
        f"Suffix:   {args.suffix}",
        f"--- Config ---",
        f"Alpha: {cfg.alpha}   Reactive: {cfg.reactive}   Path-relinking: {args.path_relinking}",
        f"Lambda Central: {args.lambda_central} (k={args.k_central})",
        f"Lambda Congest: {args.lambda_congest}",
        f"Seed: {cfg.seed}",
        f"Stop: max_iters={args.max_iters}, max_time={args.max_time}, max_no_improv={args.max_no_improv}",
        f"--- Results ---",
        f"Iters:    {iters}",
        f"Best makespan:  {best.makespan:.6f}   Time(s): {elapsed:.3f}",
    ]

    # report no stdout
    for line in report_lines:
        print(line)

    # Use o sufixo no nome do 'report.txt'
    with open(out_path / f"report{args.suffix}.txt", "w", encoding="utf-8") as r:
        r.write("\n".join(report_lines) + "\n")

    if args.save_tree:
        # Use o sufixo no nome do '.tree'
        tree_path = out_path / f"{inst.name}{args.suffix}.grasp.tree"
        with open(tree_path, "w", encoding="utf-8") as w:
            w.write(f"NAME: {inst.name}\nTYPE: TREE\nROOT: {args.root}\n")
            w.write("PARENT_SECTION (1-based: node parent)\n")
            for v, pv in enumerate(best.parent):
                pb = 0 if pv == -1 else (pv + 1)
                w.write(f"{v+1} {pb}\n")
        print(f"Saved tree: {tree_path}")

    if args.save_csv:
        # Use o sufixo no nome do '.csv'
        csv_path = out_path / f"{inst.name}{args.suffix}.grasp.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as w:
            cw = csv.writer(w)
            cw.writerow(["node", "parent", "time"])
            for v in range(inst.n()):
                pv = -1 if best.parent[v] == -1 else best.parent[v] + 1
                cw.writerow([v + 1, pv, f"{best.t[v]:.6f}"])
        print(f"Saved CSV:  {csv_path}")

if __name__ == "__main__":
    main()
