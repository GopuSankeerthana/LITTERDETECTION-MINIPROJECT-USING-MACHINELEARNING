#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  bash setup.sh
fi

if [ ! -f "litter_inference_graph/frozen_inference_graph.pb" ] || \
   [ "$(stat -f%z litter_inference_graph/frozen_inference_graph.pb 2>/dev/null || stat -c%s litter_inference_graph/frozen_inference_graph.pb)" -lt 1000000 ]; then
  bash download_model.sh
fi

./venv/bin/python app.py
