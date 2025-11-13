#!/usr/bin/env bash
# ============================================================================
# Script de varredura de parâmetros para instâncias do FTP
# ============================================================================
set -euo pipefail

# --- Configuração Base ---
ROOT=1
SEED=123
MAX_ITERS=10000
MAX_TIME=$((30*60))   # 30 minutos
NO_IMPROV=400

DEFAULT_INSTANCES_ROOT="instances_tsp"
MAX_PROCS=6           # número máximo de processos (cores) em paralelo

# --- Monta a lista de instâncias ---
declare -a INSTANCES

if [[ $# -gt 0 ]]; then
    ARG="$1"
    if [[ -f "$ARG" ]]; then
        # Usuário passou um arquivo .tsp específico
        INSTANCES=("$ARG")
    elif [[ -d "$ARG" ]]; then
        # Usuário passou um diretório: procura .tsp nele
        mapfile -t INSTANCES < <(find "$ARG" -maxdepth 2 -type f -name '*.tsp' | sort)
    else
        echo "Erro: argumento '$ARG' não é um arquivo .tsp nem um diretório válido."
        exit 1
    fi
else
    # Sem argumentos: usa o diretório padrão
    mapfile -t INSTANCES < <(find "$DEFAULT_INSTANCES_ROOT" -maxdepth 2 -type f -name '*.tsp' | sort)
fi

if [[ ${#INSTANCES[@]} -eq 0 ]]; then
    echo "Nenhuma instância .tsp encontrada."
    exit 1
fi

# --- Função para bloquear até haver slot livre (< MAX_PROCS jobs) ---
wait_for_slot() {
    while (( $(jobs -r -p | wc -l) >= MAX_PROCS )); do
        sleep 1
    done
}

# --- Função para rodar um experimento para a instância atual ---
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
        --auto-root \
        "${extra_args[@]}"

    echo "--- Experimento $suffix concluído ---"

    # Pós-processamento: extrai dados do report individual
    if [[ -f "$report_file" ]]; then
        local data_line
        data_line=$(grep "Best makespan:" "$report_file" | awk '{print $3 "," $5}')
        echo "$suffix,$data_line" >> "$CENTRAL_REPORT"
    else
        echo "AVISO: $report_file não foi encontrado. Não foi possível adicionar ao sumário."
    fi
}

# --- Loop sobre todas as instâncias ---
for INSTANCE_FILE in "${INSTANCES[@]}"; do
    echo "================================================================="
    echo "Agendando experimentos para instância: $INSTANCE_FILE"

    INSTANCE_NAME_WITH_EXT=$(basename "$INSTANCE_FILE")
    INSTANCE_NAME="${INSTANCE_NAME_WITH_EXT%.*}"
    RESULTS_DIR="results/$INSTANCE_NAME"
    CENTRAL_REPORT="$RESULTS_DIR/_summary_report.csv"

    mkdir -p "$RESULTS_DIR"

    echo "Configuracao,Makespan,Tempo_s" > "$CENTRAL_REPORT"
    echo "Relatório central será salvo em: $CENTRAL_REPORT"

    # -------------------------------------------------------------
    # Lista de Experimentos para ESTA instância
    # (todos vão para a fila global, respeitando MAX_PROCS)
    # -------------------------------------------------------------

    # 1) Guloso (reativo) (0 0 0) --reactive
    wait_for_slot
    run_exp 0.0 0.0 0.0 "_reactive_a0.00" --reactive &

    # 2) Alfa 1 (0 0 0.05)
    wait_for_slot
    run_exp 0.0 0.0 0.05 "_base_a0.05" &

    # 3) Alfa 2 (0 0 0.1)
    wait_for_slot
    run_exp 0.0 0.0 0.10 "_base_a0.10" &

    # 4) Busca local 2-opt (0 0 0.05) --use-2opt --path-relinking
    wait_for_slot
    run_exp 0.0 0.0 0.05 "_a0.05_2opt_pr" --use-2opt --path-relinking &

    # 5) Com “cong” (0.1 0 0.05) --use-2opt --path-relinking
    wait_for_slot
    run_exp 0.1 0.0 0.05 "_a0.05_2opt_pr_centr0.1" --use-2opt --path-relinking &

    # 6) Com “centralidade” (0 0.2 0.05) --use-2opt --path-relinking
    wait_for_slot
    run_exp 0.0 0.2 0.05 "_a0.05_2opt_pr_cong0.2" --use-2opt --path-relinking &

    # 7) Com os dois (0.2 0.5 0.05) --use-2opt --path-relinking
    wait_for_slot
    run_exp 0.2 0.5 0.05 "_a0.05_2opt_pr_centr0.2_cong0.5" --use-2opt --path-relinking &

    echo "Experimentos desta instância foram enfileirados: $INSTANCE_NAME"
    echo "================================================================="
done

# Espera TODOS os jobs (todas instâncias / todos experimentos) terminarem
wait

echo "Todos os experimentos concluídos para todas as instâncias."
