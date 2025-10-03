#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-COSC591-Plant}"
APP_PY="${APP_PY:-COSC591-plant-classifier.py}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.lock.txt}"
PY_VER="${PY_VER:-3.10}"
MODEL_PATH="${MODEL_PATH:-models/resnet50/resnet50_stage2_conv5.h5}"

echo "==> Bootstrap for ${ENV_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

have_conda() { command -v conda >/dev/null 2>&1; }

if ! have_conda; then
  echo "==> Conda not found. Installing Miniconda to \$HOME/miniconda3 ..."
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "${OS}-${ARCH}" in
    Linux-x86_64)  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
    Linux-aarch64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
    Darwin-x86_64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
    Darwin-arm64)  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
    *) echo "ERROR: Unsupported platform ${OS}-${ARCH}. Install Miniconda manually."; exit 1 ;;
  esac

  TMP=${TMPDIR:-/tmp}
  INST="${TMP}/miniconda_installer.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$INST"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$INST" "$URL"
  else
    echo "ERROR: need curl or wget to download Miniconda."; exit 1
  fi

  bash "$INST" -b -p "$HOME/miniconda3"
  rm -f "$INST"

  . "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  if ! have_conda; then
    [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && . "$HOME/miniconda3/etc/profile.d/conda.sh"
    [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && . "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
fi

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda still not available"; exit 1; }
eval "$(conda shell.bash hook)"


if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda env '${ENV_NAME}' already exists."
else
  echo "==> Creating env '${ENV_NAME}' (Python ${PY_VER})..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi


conda activate "${ENV_NAME}"
python -m pip install --upgrade pip


if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  if [[ "${SKIP_TORCH:-0}" == "1" ]]; then
    echo "==> Installing from ${REQUIREMENTS_FILE} (torch-related packages filtered out) ..."
    TMP_REQ="$(mktemp)"
    grep -viE '^(torch|torchvision|torchaudio|triton|pytorch(-cuda)?|nvidia-)' "${REQUIREMENTS_FILE}" > "${TMP_REQ}" || true
    python -m pip install -r "${TMP_REQ}"
    rm -f "${TMP_REQ}"
  else
    echo "==> Installing pip packages from ${REQUIREMENTS_FILE} ..."
    python -m pip install -r "${REQUIREMENTS_FILE}"
  fi
else
  echo "WARNING: ${REQUIREMENTS_FILE} not found. Installing a minimal baseline (no PyTorch)..."
  python -m pip install \
    "numpy>=1.26,<2.2" \
    "pillow>=10.3,<12" \
    "matplotlib>=3.8,<3.11" \
    "opencv-python>=4.9,<5"

fi


if [[ ! -f "${MODEL_PATH}" ]]; then
  if compgen -G "${MODEL_PATH}.part."* >/dev/null; then
    echo "==> Reassembling model from split parts: ${MODEL_PATH}.part.*"
    cat "${MODEL_PATH}.part."* > "${MODEL_PATH}.tmp"
    mv -f "${MODEL_PATH}.tmp" "${MODEL_PATH}"
    echo "==> Model reassembled at ${MODEL_PATH}"
  else
    echo "ERROR: ${MODEL_PATH} not found and no parts ${MODEL_PATH}.part.* present."
    exit 1
  fi
fi


if [[ ! -f "${APP_PY}" ]]; then
  echo "ERROR: ${APP_PY} not found in $(pwd)."
  echo "Set APP_PY=/path/to/app.py when invoking, e.g.:"
  echo "  APP_PY='/full/path/COSC591-plant-classifier.py' ./setup_and_run.sh"
  exit 1
fi

echo "==> Launching: ${APP_PY}"
python "${APP_PY}"
