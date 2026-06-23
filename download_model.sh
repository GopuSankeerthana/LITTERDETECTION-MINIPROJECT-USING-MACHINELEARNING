#!/bin/bash
set -e
cd "$(dirname "$0")"

MODEL_DIR="litter_inference_graph"
MODEL_FILE="$MODEL_DIR/frozen_inference_graph.pb"
MODEL_URL="https://media.githubusercontent.com/media/isaychris/litter-detection-tensorflow/master/litter_inference_graph/frozen_inference_graph.pb"
EXPECTED_SIZE=415611168

if [ -f "$MODEL_FILE" ]; then
  size=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE")
  if [ "$size" -eq "$EXPECTED_SIZE" ]; then
    echo "Model already downloaded ($(du -h "$MODEL_FILE" | cut -f1))."
    exit 0
  fi
  echo "Removing incomplete model ($size bytes, expected $EXPECTED_SIZE)..."
  rm -f "$MODEL_FILE"
fi

echo "Downloading trained litter detection model (~396 MB)..."
mkdir -p "$MODEL_DIR"
curl -L --progress-bar -o "$MODEL_FILE.tmp" "$MODEL_URL"
size=$(stat -f%z "$MODEL_FILE.tmp" 2>/dev/null || stat -c%s "$MODEL_FILE.tmp")
if [ "$size" -ne "$EXPECTED_SIZE" ]; then
  echo "Download failed: got $size bytes, expected $EXPECTED_SIZE"
  rm -f "$MODEL_FILE.tmp"
  exit 1
fi
mv "$MODEL_FILE.tmp" "$MODEL_FILE"
echo "Download complete: $MODEL_FILE"
