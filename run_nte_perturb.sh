#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GTR_MODEL="GTRNTE"

exec bash "${script_dir}/run_perturb.sh" "$@"
