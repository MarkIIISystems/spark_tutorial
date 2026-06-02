#!/usr/bin/env bash
set -euo pipefail

NAME="open-webui"
NAME2="searxng"
IMAGE="ghcr.io/open-webui/open-webui:v0.9.5-ollama"

cleanup() {
  echo "Signal received; stopping ${NAME}..."
  echo "Signal received; stopping ${NAME2}..."
  docker stop "${NAME}" >/dev/null 2>&1 || true
  docker stop "${NAME2}" >/dev/null 2>&1 || true
  exit 0
}
trap cleanup INT TERM HUP QUIT EXIT

# Ensure Docker CLI and daemon are available
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon not reachable." >&2
  exit 1
fi

# Already running OpenWebUI?
if [ -n "$(docker ps -q --filter "name=^${NAME}$" --filter "status=running")" ]; then
  echo "Container ${NAME} is already running."
else
  # Exists but stopped? Start it.
  if [ -n "$(docker ps -aq --filter "name=^${NAME}$")" ]; then
    echo "Starting existing container ${NAME}..."
    docker start "${NAME}" >/dev/null

  else
    # Not present: create and start it.
    echo "Creating and starting ${NAME}..."
    docker run -d -p 12000:8080 --gpus=all \
      -v open-webui:/app/backend/data \
      -v open-webui-ollama:/root/.ollama \
      --name "${NAME}" "${IMAGE}" >/dev/null
    
  fi
fi

# Already running Searxng?
if [ -n "$(docker ps -q --filter "name=^${NAME2}$" --filter "status=running")" ]; then
  echo "Container ${NAME2} is already running."
else
  # Exists but stopped? Start it.
  if [ -n "$(docker ps -aq --filter "name=^${NAME2}$")" ]; then
    echo "Starting existing container ${NAME2}..."
    docker start "${NAME2}" >/dev/null
  else
    # Not present: create and start it.
    cd ~/searxng
    echo "Creating and starting ${NAME2}..."
    docker run --name searxng -d \
    -p 8888:8080 \
    -v "./config/:/etc/searxng/" \
    -v "./data/:/var/cache/searxng/" \
    docker.io/searxng/searxng:latest
  fi
fi

echo "Running. Press Ctrl+C to stop ${NAME}."
# Keep the script alive until a signal arrives
while :; do sleep 86400; done
