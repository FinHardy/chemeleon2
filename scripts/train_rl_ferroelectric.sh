#!/usr/bin/env bash
# Train the ferroelectric-targeted RL policy (Phase 1).
#
# Two-stage training:
#   Stage 1 — warmup with base DNG rewards to establish structural validity.
#   Stage 2 — FE fine-tuning with polar symmetry, polar distortion, and
#              FE chemistry rewards added on top.
#
# Usage:
#   ./scripts/train_rl_ferroelectric.sh            # default settings
#   WARMUP_STEPS=0 ./scripts/...                   # skip warmup
#   DATASET=alex_mp_20 ./scripts/...               # use larger dataset
#   DEVICES=2 ./scripts/...                        # multi-GPU

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------
WARMUP_STEPS=${WARMUP_STEPS:-1000}
MAX_STEPS=${MAX_STEPS:-5000}
DATASET=${DATASET:-mp_20}           # mp_20 | alex_mp_20
DEVICES=${DEVICES:-1}
SEED=${SEED:-0}

echo "========================================"
echo " Ferroelectric RL Training (Phase 1)"
echo "========================================"
echo "  Dataset:       ${DATASET}"
echo "  Warmup steps:  ${WARMUP_STEPS}"
echo "  FE steps:      ${MAX_STEPS}"
echo "  Devices:       ${DEVICES}"
echo "========================================"


# ------- Move into project root (assumes script is in scripts/) -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

# ---------------------------------------------------------------------------
# Pre-flight: benchmark assets
# ---------------------------------------------------------------------------
ASSETS_DIR="benchmarks/assets"
STRUCTURES_FILE="${ASSETS_DIR}/mp_20_all_structure.json.gz"

if [ ! -f "${STRUCTURES_FILE}" ]; then
    echo ""
    echo "--- Preparing benchmark assets (first-time setup) ---"
    uv run python scripts/prepare_assets.py
fi

# ---------------------------------------------------------------------------
# Stage 1: warmup with base DNG rewards (no EnergyReward)
# ---------------------------------------------------------------------------
WARMUP_CKPT=""

if [ "${WARMUP_STEPS}" -gt 0 ]; then
    echo ""
    echo "--- Stage 1: warmup (${WARMUP_STEPS} steps) ---"

    uv run python src/train_rl.py \
        experiment="${DATASET}/rl_fe_warmup" \
        trainer.max_steps="${WARMUP_STEPS}" \
        trainer.devices="${DEVICES}" \
        seed="${SEED}" \
        task_name=rl_fe_warmup \
        logger.wandb.name="rl_fe_warmup_${DATASET}"

    # Locate the most recent warmup checkpoint
    WARMUP_CKPT=$(find logs/rl_fe_warmup/runs -name "last.ckpt" \
        -printf "%T@ %p\n" 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "${WARMUP_CKPT}" ]; then
        echo "Warning: warmup checkpoint not found — starting Stage 2 from scratch."
    else
        echo "Warmup checkpoint: ${WARMUP_CKPT}"
    fi
else
    echo "Warmup skipped (WARMUP_STEPS=0)."
fi

# ---------------------------------------------------------------------------
# Stage 2: FE-targeted fine-tuning
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 2: FE fine-tuning (${MAX_STEPS} steps) ---"

CKPT_ARG=""
if [ -n "${WARMUP_CKPT}" ]; then
    CKPT_ARG="ckpt_path=${WARMUP_CKPT}"
fi


uv run python src/train_rl.py \
    experiment="${DATASET}/rl_ferroelectric" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.devices="${DEVICES}" \
    seed="${SEED}" \
    task_name="rl_ferroelectric_${DATASET}" \
    logger.wandb.name="rl_ferroelectric_${DATASET}" \
    ${CKPT_ARG}

echo ""
echo "Training complete."
echo "Checkpoints and logs: logs/rl_ferroelectric_${DATASET}/"
