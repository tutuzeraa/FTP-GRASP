#!/usr/bin/env bash
# ============================================================================
# Script de varredura de parâmetros para UMA instância do FTP
# ============================================================================
set -euo pipefail

# --- Configuração Base ---
INSTANCE_FILE="instances_teste/literatura/d198.tsp" # A instância que vamos testar
ROOT=1
SEED=123
MAX_ITERS=3000
MAX_TIME=10
NO_IMPROV=400

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

    shift 4
    local extra_args=("$@")

    local report_file="$RESULTS_DIR/report${suffix}.txt"

    echo "--- Executando Experimento: $suffix ---"
    
    python3 src/grasp_ftp.py "$INSTANCE_FILE" \
        --root "$ROOT" --seed "$SEED" \
        --alpha "$alpha" \
        --lambda-central "$lambda_c" \
        --lambda-congest "$lambda_k" \
        --max-iters "$MAX_ITERS" --max-time "$MAX_TIME" --max-no-improv "$NO_IMPROV" \
        --save-csv \
        --suffix "$suffix" \
        "${extra_args[@]}"

    
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

# Experimento 5: alpha = 0 (guloso), sem heurísticas de score
run_exp 0.0 0.0 0 "_base_a0.0_use2opt" --use-2opt

# Experimento 6
run_exp 0.0 0.0 0.05 "_base_a0.05_use2opt" --use-2opt
# Experimento 7: 
run_exp 0.0 0.0 0.1 "_base_a0.1_use2opt" --use-2opt

# Experimento 8: 
run_exp 0.0 0.0 0.15 "_base_a0.15_use2opt" --use-2opt

# Experimento 9: alpha = 1 (totalmente aleatório), sem heurísticas de score
run_exp 0.0 0.0 1.0 "_base_a1.0_use2opt" --use-2opt

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

# Experimento 2: Apenas Congestionamento
run_exp 0.0 0.1 0.1 "_cong_0.1"

# Experimento 2: Apenas Congestionamento
run_exp 0.0 0.2 0.1 "_cong_0.2"

# Experimento 2: Apenas Congestionamento
run_exp 0.0 0.3 0.1 "_cong_0.3"

# Experimento 3: Apenas Centralidade
run_exp 0.1 0.0 0.1 "_centr_0.1"

# Experimento 2: Apenas centralidade
run_exp 0.2 0.0 0.1 "_centr_0.2"

# Experimento 2: Apenas Centralidade
run_exp 0.3 0.0 0.1 "_centr_0.3"

# Experimento 2: Apenas Centralidade
run_exp 0.5 0.0 0.1 "_centr_0.5"

# Experimento 2: Apenas Centralidade
run_exp 1.0 0.0 0.1 "_centr_0.5"

# Experimento 4: Centralidade e congestionamento original
run_exp 0.1 0.1 0.1 "_centr_cong_0.1"

# Experimento 4: Centralidade e congestionamento original
run_exp 0.2 0.2 0.1 "_centr_cong_0.2"

# Experimento 4: Centralidade e congestionamento original
run_exp 0.3 0.3 0.1 "_centr_cong_0.3"

# Experimento 4: Centralidade e congestionamento original
run_exp 0.2 0.5 0.1 "_centr_cong_0.2_0.5"

# Experimento 4: Centralidade e congestionamento original
run_exp 0.5 0.2 0.1 "_centr_cong_0.5_0.2"

# Experimento 5: alpha = 0 (guloso), sem heurísticas de score
run_exp 0.0 0.0 0 "_base_a0.0_reactive" --reactive

# Experimento 6
run_exp 0.0 0.0 0.05 "_base_a0.05_reactive" --reactive
# Experimento 7: 
run_exp 0.0 0.0 0.1 "_base_a0.1_reactive" --reactive

# Experimento 8: 
run_exp 0.0 0.0 0.15 "_base_a0.15_reactive" --reactive

# Experimento 9: alpha = 1 (totalmente aleatório), sem heurísticas de score
run_exp 0.0 0.0 1.0 "_base_a1.0_reactive" --reactive

# Experimento 5: alpha = 0 (guloso), sem heurísticas de score
run_exp 0.0 0.0 0 "_base_a0.0_pathRelinking" --path-relinking

# Experimento 6
run_exp 0.0 0.0 0.05 "_base_a0.05_pathRelinking" --path-relinking
# Experimento 7: 
run_exp 0.0 0.0 0.1 "_base_a0.1_pathRelinking" --path-relinking

# Experimento 8: 
run_exp 0.0 0.0 0.15 "_base_a0.15_pathRelinking" --path-relinking

# Experimento 9: alpha = 1 (totalmente aleatório), sem heurísticas de score
run_exp 0.0 0.0 1.0 "_base_a1.0_pathRelinking" --path-relinking

echo "Todos os experimentos concluídos. Relatório central salvo em: $CENTRAL_REPORT"