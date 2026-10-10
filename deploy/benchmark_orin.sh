#!/bin/bash
# ==============================================================================
# Jetson AGX Orin Deployment & Benchmark Procedure
# Target: MaCVi @ WACV 2027 Track 2 (EURS Challenge)
# Model: EURS-Net V2 (Pilot Epoch 2 Checkpoint)
# Static Input: 1 x 3 x 1024 x 1024 FP16/FP32
# Outputs: restored_image (1 x 3 x 1024 x 1024), seg_logits (1 x 5 x 1024 x 1024)
# ==============================================================================

set -e

ONNX_MODEL="deploy/exports/pilot_epoch_2/eurs_net_v2_pilot_ep2.onnx"
ENGINE_FP16="deploy/exports/pilot_epoch_2/eurs_net_v2_pilot_ep2_fp16.engine"
TIMING_JSON="deploy/exports/pilot_epoch_2/orin_benchmark_results.json"

echo "=============================================================================="
echo "    JETSON AGX ORIN BENCHMARK PROCEDURE: EURS-NET V2"
echo "=============================================================================="

if ! command -v trtexec &> /dev/null; then
    echo "[ERROR] trtexec not found! Please ensure NVIDIA TensorRT is in PATH."
    echo "Expected location on JetPack: /usr/src/tensorrt/bin/trtexec"
    exit 1
fi

if [ ! -f "$ONNX_MODEL" ]; then
    echo "[ERROR] ONNX model missing: $ONNX_MODEL"
    exit 1
fi

echo "[1/3] Setting Jetson Clocks to Maximum Performance Mode..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0  # Mode 0: MAXN (unconstrained power mode on AGX Orin)
fi
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
fi

echo "[2/3] Building TensorRT FP16 Engine via trtexec..."
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$ENGINE_FP16" \
    --fp16 \
    --explicitBatch \
    --minShapes=input:1x3x1024x1024 \
    --optShapes=input:1x3x1024x1024 \
    --maxShapes=input:1x3x1024x1024 \
    --workspace=4096 \
    --warmUp=200 \
    --iterations=1000 \
    --duration=10 \
    --dumpProfile \
    --exportTimes="$TIMING_JSON"

echo "[3/3] Benchmark Run Complete!"
echo "Engine saved to: $ENGINE_FP16"
echo "Timing profile saved to: $TIMING_JSON"
