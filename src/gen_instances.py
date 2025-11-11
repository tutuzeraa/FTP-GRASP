import argparse
from pathlib import Path
import math

def circle_layout(n, R=100.0, start_angle=0.0):
    pts = []
    for i in range(n):
        ang = start_angle + 2.0 * math.pi * (i / n)
        pts.append((R * math.cos(ang), R * math.sin(ang)))
    return pts

def write_tsplib_euc2d(out_path: Path, name: str, coords):
    n = len(coords)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            # Muitos parsers aceitam float; seu parse deve aceitar também.
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")

def write_csv_xy(out_path: Path, coords):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for (x, y) in coords:
            f.write(f"{x:.6f},{y:.6f}\n")


def gen_star(n: int, radius=100.0):
    """
    Raiz = nó 1 no (0,0); folhas em círculo de raio 'radius'.
    Compatível com EUC_2D (distâncias euclidianas).
    """
    assert n >= 2
    coords = [(0.0, 0.0)]
    if n > 1:
        leaves = circle_layout(n - 1, R=radius, start_angle=0.0)
        coords.extend(leaves)
    return coords

def gen_path(n: int, step=10.0):
    """
    Nós ao longo do eixo x: (0,0), (step,0), (2*step,0), ...
    """
    assert n >= 2
    return [(i * step, 0.0) for i in range(n)]

def gen_hypercube_embed(d: int, scale=10.0):
    """
    Embedding 2D do hipercubo Q_d: divide os bits entre (x,y).
    Produz uma grade 2D de tamanho (2^dx) x (2^dy), dx+dy=d.
    É um embedding euclidiano (não preserva Hamming exato).
    """
    assert d >= 1
    dx = d // 2
    dy = d - dx
    nx = 1 << dx
    ny = 1 << dy
    coords = []
    # Mapeia índice i in [0, 2^d) para (x,y) na grade
    for i in range(1 << d):
        x_bits = i & ((1 << dx) - 1)
        y_bits = i >> dx
        x = x_bits * scale
        y = y_bits * scale
        coords.append((x, y))
    return coords

def gen_debruijn_embed(k: int, m: int = 2, radius=100.0):
    """
    Embedding euclidiano do grafo de De Bruijn: apenas posiciona N=m^k pontos em círculo.
    (Não é métrica de menor caminho; é só geometria compatível com EUC_2D.)
    """
    assert k >= 1 and m >= 2
    N = m ** k
    return circle_layout(N, R=radius, start_angle=0.0)

# =========================
# Main / CLI
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Gerador de instâncias analíticas em EUC_2D/CSV para 'instances_analytic/'."
    )
    ap.add_argument("--outdir", default="instances_analytic", help="Pasta de saída (default: instances_analytic)")
    ap.add_argument("--format", choices=["tsp", "csv", "both"], default="tsp",
                    help="Formato de saída (default: tsp)")
    # Conjuntos padrão
    ap.add_argument("--stars", default="8,16,32,64", help="Lista de n para estrelas (ex: 8,16,32)")
    ap.add_argument("--paths", default="8,16,32,64", help="Lista de n para paths (ex: 8,16,32)")
    ap.add_argument("--hypercubes", default="4,5,6", help="Lista de d para hypercube_embed (ex: 4,5,6)")
    ap.add_argument("--debruijns", default="6,2;7,2",
                    help="Lista de k,m para debruijn_embed separados por ';' (ex: 6,2;7,2)")
    # Parâmetros geométricos
    ap.add_argument("--radius", type=float, default=100.0, help="Raio para star/debruijn (default: 100)")
    ap.add_argument("--step", type=float, default=10.0, help="Passo para path (default: 10)")
    ap.add_argument("--scale", type=float, default=10.0, help="Escala da grade para hypercube_embed (default: 10)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    want_tsp = args.format in ("tsp", "both")
    want_csv = args.format in ("csv", "both")

    def dump(name, coords):
        if want_tsp:
            write_tsplib_euc2d(outdir / f"{name}.tsp", name, coords)
        if want_csv:
            write_csv_xy(outdir / f"{name}.csv", coords)

    # Estrelas
    if args.stars.strip():
        for n_str in args.stars.split(","):
            n = int(n_str.strip())
            coords = gen_star(n, radius=args.radius)
            dump(f"star_n{n}", coords)

    # Paths (linhas)
    if args.paths.strip():
        for n_str in args.paths.split(","):
            n = int(n_str.strip())
            coords = gen_path(n, step=args.step)
            dump(f"path_n{n}", coords)

    # Hypercube (embedding 2D)
    if args.hypercubes.strip():
        for d_str in args.hypercubes.split(","):
            d = int(d_str.strip())
            coords = gen_hypercube_embed(d, scale=args.scale)
            dump(f"hypercube_embed_d{d}_n{1<<d}", coords)

    # De Bruijn (embedding 2D)
    if args.debruijns.strip():
        for item in args.debruijns.split(";"):
            item = item.strip()
            if not item:
                continue
            k_str, m_str = item.split(",")
            k, m = int(k_str), int(m_str)
            coords = gen_debruijn_embed(k=k, m=m, radius=args.radius)
            dump(f"debruijn_embed_k{k}_m{m}_n{m**k}", coords)

    print(f"[ok] Instâncias geradas em {outdir.resolve()}")
    # lista curta
    for p in sorted(outdir.glob("*")):
        if p.suffix in (".tsp", ".csv"):
            print(" -", p.name)

if __name__ == "__main__":
    main()
