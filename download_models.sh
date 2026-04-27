#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://lab.local:5000/api/v1/projects/car-racing-q-learning"
DEST="./downloaded_models"

mkdir -p "$DEST"

echo "Downloading latest checkpoint from $BASE_URL..."
curl -f -o "$DEST/q_model.pt" "$BASE_URL/files/checkpoints/q_model.pt"
echo "Saved: $DEST/q_model.pt"
