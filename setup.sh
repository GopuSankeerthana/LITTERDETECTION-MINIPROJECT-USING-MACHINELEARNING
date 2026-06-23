#!/bin/bash
set -e
cd "$(dirname "$0")"

chmod +x download_model.sh run.sh

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./download_model.sh

echo ""
echo "Setup complete. Run the app with:"
echo "  ./run.sh"
