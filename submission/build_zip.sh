#!/usr/bin/env bash
# Copyright (c) 2026 AIMS Foundations. MIT License.
#
# Build a flat Codabench-ready submission ZIP for CAIMIRA+EB.
#
# Why an EXPLICIT allowlist instead of `zip -r ./submission.zip submission/`:
#   - The kit pre-validator rejects ZIPs with model.py inside a subdirectory
#     (`starting_kit/tools/check_submission_zip.py:56-58`); flat layout required.
#   - The kit pre-validator rejects ZIPs containing other ZIPs and absolute
#     paths (`:54`, `:79-83`).
#   - The "obvious" recipe (`zip -r ... -x '*.pyc' '__pycache__/*'`) silently
#     slurps per-seed audit copies of .pt files, blowing past the kit's
#     single-.pt-per-submission expectation (see the parent repo's
#     `scripts/check_submission.sh:101-116`).
# An explicit allowlist sidesteps all three failure modes.
#
# Usage:
#   bash submission/build_zip.sh                 # writes submission_caimira.zip
#   bash submission/build_zip.sh out.zip         # custom output path
#   bash submission/build_zip.sh --force         # overwrite existing ZIP
#
# Refuses to overwrite an existing ZIP unless --force is passed.
# Refuses to build if caimira_lite.pt / eb_tables.json / caimira_lite.meta.json
# are missing — run `python submission/train.py` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="${SCRIPT_DIR}"

FORCE=0
OUTPUT_ZIP=""
for arg in "$@"; do
  case "${arg}" in
    --force|-f) FORCE=1 ;;
    -h|--help)
      sed -n '3,25p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      if [[ -n "${OUTPUT_ZIP}" ]]; then
        echo "build_zip.sh: unexpected extra argument: ${arg}" >&2
        exit 2
      fi
      OUTPUT_ZIP="${arg}"
      ;;
  esac
done

if [[ -z "${OUTPUT_ZIP}" ]]; then
  OUTPUT_ZIP="$(cd "${SUBMISSION_DIR}/.." && pwd)/submission_caimira.zip"
fi

# Required files (artifact contract for the CAIMIRA submission mode).
REQUIRED_FILES=(
  "model.py"
  "labeling.py"
  "models.txt"
  "caimira_lite.py"
  "caimira_lite.pt"
  "caimira_lite.meta.json"
  "eb_tables.json"
)

# Refuse if any required file is missing.
missing=()
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${SUBMISSION_DIR}/${f}" ]]; then
    missing+=("${f}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "build_zip.sh: missing required files in ${SUBMISSION_DIR}:" >&2
  for m in "${missing[@]}"; do
    echo "  - ${m}" >&2
  done
  echo "" >&2
  echo "Run 'python ${SUBMISSION_DIR}/train.py' first to produce" >&2
  echo "caimira_lite.pt / caimira_lite.meta.json / eb_tables.json." >&2
  exit 1
fi

if [[ -f "${OUTPUT_ZIP}" && "${FORCE}" -ne 1 ]]; then
  echo "build_zip.sh: refusing to overwrite ${OUTPUT_ZIP} (pass --force to override)" >&2
  exit 1
fi

# Single-.pt safety check: confirm only one .pt file is being shipped.
# Mirrors the parent repo's scripts/check_submission.sh:101-116 enforcement.
pt_count=0
for f in "${REQUIRED_FILES[@]}"; do
  if [[ "${f}" == *.pt ]]; then
    pt_count=$((pt_count + 1))
  fi
done
if (( pt_count != 1 )); then
  echo "build_zip.sh: allowlist contains ${pt_count} .pt files; expected exactly 1" >&2
  exit 1
fi

rm -f "${OUTPUT_ZIP}"

# `zip -j` strips paths so model.py lands at the ZIP root (the kit
# pre-validator's L56-58 requirement).
(
  cd "${SUBMISSION_DIR}"
  zip -j "${OUTPUT_ZIP}" "${REQUIRED_FILES[@]}"
) >/dev/null

# Post-build sanity: verify the ZIP is flat (no subdirectories).
nested=$(unzip -l "${OUTPUT_ZIP}" | awk 'NR > 3 && /\// {print $NF}' | head -1 || true)
if [[ -n "${nested}" ]]; then
  echo "build_zip.sh: ZIP contains nested paths (${nested}); aborting" >&2
  rm -f "${OUTPUT_ZIP}"
  exit 1
fi

# Post-build sanity: confirm exactly the allowlisted files landed.
zip_files=$(unzip -l "${OUTPUT_ZIP}" | awk 'NR > 3 && NF >= 4 {print $NF}' | grep -v '^$' | grep -v '^-' || true)
allow_csv=$(printf '%s\n' "${REQUIRED_FILES[@]}" | sort)
actual_csv=$(echo "${zip_files}" | sort)
if [[ "${allow_csv}" != "${actual_csv}" ]]; then
  echo "build_zip.sh: ZIP file list differs from allowlist" >&2
  echo "  expected: ${allow_csv//$'\n'/ , }" >&2
  echo "  actual:   ${actual_csv//$'\n'/ , }" >&2
  exit 1
fi

size_bytes=$(wc -c <"${OUTPUT_ZIP}" | tr -d ' ')
echo "build_zip.sh: built ${OUTPUT_ZIP} (${size_bytes} bytes, ${#REQUIRED_FILES[@]} files)"
echo "build_zip.sh: contents:"
unzip -l "${OUTPUT_ZIP}" | sed 's/^/  /'
echo ""
echo "Next steps:"
echo "  1. Validate locally with the kit's check_submission_zip.py"
echo "     (predictive-eval-competition/starting_kit/tools/check_submission_zip.py)."
echo "  2. Run the D-9 transfer-audit gate"
echo "     (predictive-eval-competition/llm_wiki/decisions/D9-pre-submission-stress-test-gate.md)."
echo "  3. Only then upload to Codabench competition 15934."
