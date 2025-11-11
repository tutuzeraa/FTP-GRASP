#!/usr/bin/env bash
# ============================================================================
# Script de execução em lote para o GRASP do Freeze-Tag Problem (FTP)
# - Varre instâncias .tsp (TSPLIB EUC_2D) e .csv no diretório instances/
# - Filtra e IGNORA arquivos com número de vértices > 500
# - Chama grasp_ftp.py com parâmetros padrão e salva árvore/tempos por instância
# ============================================================================
set -euo pipefail

# Diretório com as instâncias
INST_DIR="instances_tsp"

# Parâmetros do GRASP (conforme proposta)
ROOT=1          # índice 1-based da raiz para a CLI (internamente vira 0-based)
ALPHA=0.2     # tamanho efetivo da RCL (0 => guloso, 1 => aleatório)
SEED=123        # semente para reprodutibilidade
MAX_ITERS=3000  # limite de iterações
MAX_TIME=1      # limite de tempo por instância (segundos)
NO_IMPROV=400   # limite de iterações sem melhoria
MAX_N=500       # LIMITE: ignora instâncias com mais de 500 vértices

# ----------------------------------------------------------------------------
# Função: num_vertices
# Determina o número de vértices de uma instância .tsp (conta coordenadas
# entre NODE_COORD_SECTION e EOF) ou .csv (linhas não vazias).
# Compatível com BSD/macOS awk (evita variável reservada 'in').
# ----------------------------------------------------------------------------
num_vertices() {
  local f="$1"
  case "$f" in
    *.tsp|*.TSP)
      # Conta linhas de coordenadas (id x y) após NODE_COORD_SECTION
      awk '
        BEGIN { inside=0; c=0 }
        toupper($0) ~ /^NODE_COORD_SECTION/ { inside=1; next }
        toupper($0) ~ /^EOF/ { inside=0 }
        inside && NF>=3 { c++ }
        END { print c+0 }
      ' "$f"
      ;;
    *.csv|*.CSV)
      # CSV: assume uma coordenada por linha; ignora linhas vazias
      grep -v -E "^[[:space:]]*$" "$f" | wc -l | tr -d " "
      ;;
    *)
      echo 0
      ;;
  esac
}

# ----------------------------------------------------------------------------
# Loop principal: percorre arquivos suportados, filtra por MAX_N e executa
# ----------------------------------------------------------------------------
for f in "$INST_DIR"/*; do
  # Ignora não-arquivos (subpastas, etc.)
  [[ -f "$f" ]] || continue

  # Considera apenas extensões suportadas
  case "$f" in
    *.tsp|*.TSP|*.csv|*.CSV) : ;;
    *) continue ;;
  esac

  # Conta vértices e aplica filtro MAX_N
  N=$(num_vertices "$f")
  if [[ "$N" -gt "$MAX_N" ]]; then
    echo ">>> $f  (n=$N)  — ignorado (n > $MAX_N)"
    continue
  fi

  # Execução
  echo ">>> $f  (n=$N)"
  python3 src/grasp_ftp.py "$f" \
    --root "$ROOT" \
    --alpha "$ALPHA" --seed "$SEED" \
    --lambda-central 0.25 --k-central 15 \
    --lambda-congest 0.5 \
    --max-iters "$MAX_ITERS" --max-time "$MAX_TIME" --max-no-improv "$NO_IMPROV" \
    --save-tree --save-csv
done
