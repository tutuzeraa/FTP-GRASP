#!/usr/bin/env bash
# ============================================================================
# Script de varredura de parâmetros para UMA instância do FTP
# ============================================================================
set -euo pipefail

# --- Configuração Base ---
INSTANCE_FILE="instances_teste/a280.tsp" # A instância que vamos testar
ROOT=1
SEED=123
MAX_ITERS=3000
MAX_TIME=60
NO_IMPROV=400
K_CENTRAL=15

# --- Configuração dos Relatórios ---

INSTANCE_NAME_WITH_EXT=$(basename "$INSTANCE_FILE")
INSTANCE_NAME="${INSTANCE_NAME_WITH_EXT%.*}" 
RESULTS_DIR="results/$INSTANCE_NAME"
CENTRAL_REPORT="$RESULTS_DIR/_summary_report.csv"

mkdir -p "$RESULTS_DIR"

echo "Configuracao,Makespan,Tempo_s" > "$CENTRAL_REPORT"
echo "Relatório central será salvo em: $CENTRAL_REPORT"


run_exp() {
    local lambda_c="$1"
    local lambda_k="$2"
    local alpha="$3"
    local suffix="$4"

    local report_file="$RESULTS_DIR/report${suffix}.txt"

    echo "--- Executando Experimento: $suffix ---"
    
    python3 src/grasp_ftp.py "$INSTANCE_FILE" \
        --root "$ROOT" --seed "$SEED" \
        --alpha "$alpha" \
        --lambda-central "$lambda_c" --k-central "$K_CENTRAL" \
        --lambda-congest "$lambda_k" \
        --path-relinking \
        --max-iters "$MAX_ITERS" --max-time "$MAX_TIME" --max-no-improv "$NO_IMPROV" \
        --save-tree --save-csv \
        --suffix "$suffix"
    
    echo "--- Experimento $suffix concluído ---"

    # --- PÓS-PROCESSAMENTO: Extrai dados do report individual ---
    if [[ -f "$report_file" ]]; then
        local data_line=$(grep "Best makespan:" "$report_file" | awk '{print $3 "," $5}')
        echo "$suffix,$data_line" >> "$CENTRAL_REPORT"
    else
        echo "AVISO: $report_file não foi encontrado. Não foi possível adicionar ao sumário."
    fi
}

# --- Lista de Experimentos ---

# Experimento 1: Sem heurísticas de score (base)
run_exp 0.0 0.0 0.2 "_base_a0.2"

# Experimento 2: Apenas Congestionamento
run_exp 0.0 0.5 0.2 "_cong_a0.2"

# Experimento 3: Apenas Centralidade
run_exp 0.5 0.0 0.2 "_centr_a0.2"

# Experimento 4: Centralidade e congestionamento original
run_exp 0.25 0.5 0.2 "_centr_cong_a0.2"

# Experimento 5: alpha = 0 (guloso), sem heurísticas de score
run_exp 0.0 0.0 0 "_base_a0.0"

# Experimento 6
run_exp 0.0 0.0 0.05 "_base_a0.05"

# Experimento 7: 
run_exp 0.0 0.0 0.1 "_base_a0.1"

# Experimento 8: 
run_exp 0.0 0.0 0.15 "_base_a0.15"

# Experimento 9: alpha = 1 (totalmente aleatório), sem heurísticas de score
run_exp 0.0 0.0 1.0 "_base_a1.0"


echo "Todos os experimentos concluídos. Relatório central salvo em: $CENTRAL_REPORT"