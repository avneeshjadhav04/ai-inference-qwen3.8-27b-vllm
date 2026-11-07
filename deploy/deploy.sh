#!/usr/bin/env bash
# Qwen3.8-27B-FP8 + vLLM on 2x RTX 3090 — single-file reproducible deploy.
# Usage: sudo bash deploy.sh
#        ENABLE_TUNNELS=0 sudo bash deploy.sh   # LAN-only, no public tunnels
# Idempotent: safe to re-run on a fresh Ubuntu 24.04 VM or the current one.
# Exposes: vLLM OpenAI API on :8000, custom chat UI on :8080 (both 0.0.0.0, public).
#
# Prereqs: Ubuntu 24.04, root, 2x 24GB VRAM (TP=2; falls back to TP=N),
#   NVIDIA driver + nvidia-smi working, ~60GB free disk
#   (~30GB HF weights + ~10GB image + vLLM compile cache), ports 8000/8080 free.
# Benched config: vLLM v0.32.0, max-len 32768, max-seqs 128, gpu-util 0.93,
#   --reasoning-parser qwen3. See ../bench/README.md for expected numbers.
set -euo pipefail

MODEL_ID="Qwen/Qwen3.8-27B-FP8"
SERVED_NAME="Qwen3.8-27B-FP8"
# Pinned to the benched release. Digest is the multi-arch manifest list digest
# (amd64 + arm64) resolved via `docker buildx imagetools inspect`.
VLLM_IMAGE="vllm/vllm-openai:v0.32.0"
VLLM_DIGEST="sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
IMAGE_REF="${VLLM_IMAGE}@${VLLM_DIGEST}"
VLLM_PORT="8000"
UI_PORT="8080"
# Set to 0 for LAN-only deploys (no cloudflared quick tunnels).
ENABLE_TUNNELS="${ENABLE_TUNNELS:-1}"
# Heavy data must live on a native Linux filesystem with free space.
# Docker bind-mounts fail on encrypted/overlay mounts (e.g. gocryptfs /root).
# Override any of these via env, e.g. HF_CACHE=/mnt/data/hf-cache sudo -E bash deploy.sh
UI_DIR="${UI_DIR:-/opt/qwen-ui}"
HF_CACHE="${HF_CACHE:-/var/lib/qwen/hf-cache}"
# vLLM compile cache survives container recreates (else every restart costs ~20 min recompile)
VLLM_CACHE="${VLLM_CACHE:-/var/lib/qwen/vllm-cache}"
CONTAINER="qwen-vllm"
MAX_MODEL_LEN="32768"
GPU_MEM_UTIL="0.93"
TENSOR_PARALLEL="2"
# Qwen3.8 hybrid (Mamba/GDN) needs fewer concurrent seqs on 2x24GB.
# Default 256 exceeds Mamba cache blocks (~155). 128 is safe.
MAX_NUM_SEQS="128"

log() { echo "[deploy] $*"; }

# --- 0. sanity: must run as root on the VM ---
if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo bash deploy.sh)" >&2; exit 1
fi

# --- 1. GPU check ---
log "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NGPU" -lt 2 ]; then
  log "WARNING: expected 2 GPUs, found $NGPU. Continuing with TP=$NGPU."
  TENSOR_PARALLEL="$NGPU"
fi

# --- 2. Docker + NVIDIA toolkit ---
if ! command -v docker >/dev/null 2>&1; then
  log "Installing docker..."
  apt-get update && apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
  apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
if ! docker info 2>/dev/null | grep -qi "nvidia"; then
  if ! dpkg -l | grep -q nvidia-container-toolkit; then
    log "Installing nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker || true
    systemctl restart docker || service docker restart || true
    sleep 3
  fi
fi
docker --version
docker compose version || true

# --- 3. dirs (heavy data needs a native disk; overlay / is often small) ---
mkdir -p "$HF_CACHE" "$UI_DIR" "$VLLM_CACHE"
log "Disk: $(df -h / "$(dirname "$HF_CACHE")" 2>/dev/null | tail -n +2)"
# Fail fast if there is clearly not enough room for weights + image + cache.
FREE_GB=$(df --output=avail -BG "$(dirname "$HF_CACHE")" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 60 ]; then
  log "WARNING: only ${FREE_GB}GB free near $HF_CACHE; recommend 60GB+ free. Continuing anyway."
fi

# --- 4. UI file must sit next to this script as ui.html ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/ui.html" ]; then
  echo "ERROR: ui.html not found next to $0 ($SCRIPT_DIR/ui.html)" >&2
  echo "Copy both deploy.sh and ui.html to the VM first." >&2
  exit 1
fi
cp -f "$SCRIPT_DIR/ui.html" "$UI_DIR/index.html"
log "UI staged at $UI_DIR/index.html"

# --- 5. pull vLLM (pinned digest, reproducible) ---
log "Pulling $IMAGE_REF ( ~10GB, one-time )..."
docker pull "$IMAGE_REF"

# --- 6. (re)start vLLM ---
log "Starting vLLM: $MODEL_ID TP=$TENSOR_PARALLEL len=$MAX_MODEL_LEN ..."
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p "$VLLM_PORT:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$VLLM_CACHE:/root/.cache/vllm" \
  --name "$CONTAINER" \
  "$IMAGE_REF" \
  "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --allowed-origins '["*"]'

log "Waiting for vLLM (model download ~30GB on first run, can take 10-30 min)..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    log "vLLM is up after ${i}x15s."
    break
  fi
  if [ $((i % 8)) -eq 0 ]; then log "still waiting... ($((i*15))s) tail:"; docker logs --tail 5 "$CONTAINER" 2>&1 || true; fi
  sleep 15
  if [ "$i" -eq 120 ]; then log "ERROR: vLLM did not come up. Last logs:"; docker logs --tail 100 "$CONTAINER" 2>&1 || true; exit 1; fi
done

# --- 7. start UI (lightweight, no docker to save 31G overlay) ---
log "Starting UI on :$UI_PORT ..."
pkill -f "http.server $UI_PORT" 2>/dev/null || true
nohup python3 -m http.server "$UI_PORT" --bind 0.0.0.0 --directory "$UI_DIR" >"$UI_DIR/ui.log" 2>&1 &
echo $! > "$UI_DIR/ui.pid"
sleep 2
curl -sf "http://127.0.0.1:$UI_PORT/" >/dev/null && log "UI ok on :$UI_PORT" || { log "UI failed:"; cat "$UI_DIR/ui.log"; exit 1; }

# --- 8. smoke test ---
log "Smoke test chat..."
curl -s "http://127.0.0.1:$VLLM_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED_NAME\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],
  \"max_tokens\": 16,
  \"temperature\": 0,
  \"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": false}}
}" | head -c 600; echo

PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "<vm-ip>")
echo "================ DONE ================"
echo "API (local): http://127.0.0.1:$VLLM_PORT/v1  (model: $SERVED_NAME)"
echo "UI  (local): http://127.0.0.1:$UI_PORT/"
echo "Logs: docker logs -f $CONTAINER | tail -f $UI_DIR/ui.log"
echo "Re-deploy on a new VM: copy deploy.sh + ui.html, run: sudo bash deploy.sh"

# --- 9. public tunnels (OPTIONAL, INSECURE BY DEFAULT) ---
# WARNING: quick tunnels expose API + UI on public https URLs with NO AUTH.
# Anyone with the URL can use your GPU. Fine for a demo; for anything else put
# Cloudflare Access / Tailscale in front, or run LAN-only:
#   ENABLE_TUNNELS=0 sudo bash deploy.sh
# or use SSH port-forwarding instead: ssh -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 <vm>
if [ "$ENABLE_TUNNELS" = "0" ]; then
  log "ENABLE_TUNNELS=0 — skipping public tunnels (LAN-only)."
  echo "================ DONE (LAN-ONLY) ================"
  echo "API: http://<vm-ip>:$VLLM_PORT/v1  (model: $SERVED_NAME)"
  echo "UI : http://<vm-ip>:$UI_PORT/"
  exit 0
fi
# Exposes API + UI on public https URLs via cloudflared quick tunnels (no account needed).
if ! command -v cloudflared >/dev/null 2>&1; then
  log "Installing cloudflared..."
  curl -sL --max-time 120 -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
log "Starting public tunnels..."
nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$VLLM_PORT" >"$UI_DIR/tunnel-api.log" 2>&1 &
echo $! > "$UI_DIR/tunnel-api.pid"
nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$UI_PORT" >"$UI_DIR/tunnel-ui.log" 2>&1 &
echo $! > "$UI_DIR/tunnel-ui.pid"
API_URL=""; UI_URL=""
for i in $(seq 1 12); do
  API_URL=$(grep -oE "https://[A-Za-z0-9.-]+\.trycloudflare\.com" "$UI_DIR/tunnel-api.log" 2>/dev/null | sort -u | head -n 1)
  UI_URL=$(grep -oE "https://[A-Za-z0-9.-]+\.trycloudflare\.com" "$UI_DIR/tunnel-ui.log" 2>/dev/null | sort -u | head -n 1)
  [ -n "$API_URL" ] && [ -n "$UI_URL" ] && break
  sleep 5
done
# Pre-fill the UI's backend field with the public API URL so no copy-paste is needed.
if [ -n "$API_URL" ]; then
  sed -i "s|<input id=\"backend\" type=\"text\" value=\"\"|<input id=\"backend\" type=\"text\" value=\"$API_URL\"|" "$UI_DIR/index.html"
fi
echo "================ PUBLIC (NO AUTH — SHARE CAREFULLY) ================"
echo "API: $API_URL/v1  (model: $SERVED_NAME)"
echo "UI : $UI_URL/  (backend pre-filled, shows tok/sec)"
echo "NOTE: quick-tunnel URLs change on each deploy.sh run. No auth — anyone with the URL can use it."
echo "HARDEN: front with Cloudflare Access or Tailscale, or re-run with ENABLE_TUNNELS=0 + ssh -L."
