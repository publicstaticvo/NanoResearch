#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$ROOT/scripts/check_release_pack.py"
python "$ROOT/scripts/check_paper_claims.py"
