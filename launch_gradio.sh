#!/usr/bin/env bash
#
# Bootstrap helper for Auto ACR Autopilot Gradio demo (macOS/Linux).
# Detects an active environment, installs dependencies if needed,
# and launches run/gradio_app.py via a virtualenv.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADIO_APP="${ROOT}/run/gradio_app.py"
REQUIREMENTS_TXT="${ROOT}/requirements.txt"
ENVIRONMENT_YML="${ROOT}/environment.yml"
VENV_PATH="${ROOT}/.venv"

if [[ ! -f "${GRADIO_APP}" ]]; then
  echo "[error] Could not locate run/gradio_app.py. Please ensure the repository is intact."
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[error] No python interpreter found (python3/python)."
  exit 1
fi

run_cmd() {
  echo "→ $*"
  "$@"
}

ask_yes_no() {
  local prompt="$1"
  local default="$2"
  local suffix response
  if [[ "${default}" =~ ^[Yy]$ ]]; then
    suffix=" [Y/n]"
  else
    suffix=" [y/N]"
  fi
  while true; do
    read -r -p "${prompt}${suffix} " response || response=""
    if [[ -z "${response}" ]]; then
      [[ "${default}" =~ ^[Yy]$ ]] && return 0 || return 1
    fi
    case "${response}" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      [Nn]|[Nn][Oo]) return 1 ;;
    esac
    echo "Please answer yes or no."
  done
}

choose_venv_path() {
  echo "[info] Existing virtualenv directories in this repo:"
  local found=0
  for dir in "${ROOT}"/.venv* "${ROOT}"/venv*; do
    [[ -d "${dir}" ]] || continue
    found=1
    echo "  - ${dir}"
  done
  [[ ${found} -eq 0 ]] && echo "  (none detected)"
  local default_path="${VENV_PATH}"
  read -r -p "Enter virtualenv path to use [${default_path}]: " input_path || input_path=""
  input_path="${input_path:-$default_path}"
  if [[ "${input_path,,}" == "quit" ]]; then
    VENV_PATH="quit"
    return 0
  fi
  if [[ "${input_path}" != /* ]]; then
    input_path="${ROOT}/${input_path}"
  fi
  VENV_PATH="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${input_path}")"
  return 0
}

ensure_current_env() {
  if "${PYTHON_BIN}" -c "import gradio" >/dev/null 2>&1; then
    return 0
  fi
  echo "[info] Gradio not found in the active environment."
  if ! ask_yes_no "Install dependencies with pip install -r requirements.txt?" "Y"; then
    echo "[error] Cannot continue without Gradio."
    return 1
  fi
  run_cmd "${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS_TXT}"
}

ensure_virtualenv() {
  local env_path="$1"
  local system_python
  if command -v python3 >/dev/null 2>&1; then
    system_python="$(command -v python3)"
  else
    system_python="$(command -v python)"
  fi
  if [[ -x "${env_path}/bin/python" ]]; then
    echo "[info] Virtual environment already exists at ${env_path}"
    if ask_yes_no "Reinstall requirements inside the virtualenv?" "N"; then
      run_cmd "${env_path}/bin/python" -m pip install -r "${REQUIREMENTS_TXT}"
    fi
  else
    mkdir -p "$(dirname "${env_path}")"
    echo "[info] Creating virtual environment at ${env_path}"
    run_cmd "${system_python}" -m venv "${env_path}"
    echo "[info] Installing dependencies inside the virtualenv"
    run_cmd "${env_path}/bin/python" -m pip install --upgrade pip
    run_cmd "${env_path}/bin/python" -m pip install -r "${REQUIREMENTS_TXT}"
  fi
}

launch_with_interpreter() {
  echo "[info] Launching Gradio app using: $*"
  run_cmd "$@" "${GRADIO_APP}"
}

main() {
  if [[ -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[info] Detected active Python environment."
    ensure_current_env || exit 1
    launch_with_interpreter "${PYTHON_BIN}"
    exit 0
  fi

  echo "[info] No active Python environment detected."
  choose_venv_path || exit 1
  if [[ "${VENV_PATH}" == "quit" ]]; then
    echo "Aborting."
    exit 0
  fi
  ensure_virtualenv "${VENV_PATH}" || exit 1
  launch_with_interpreter "${VENV_PATH}/bin/python"
}

main "$@"
