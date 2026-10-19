#!/bin/bash
# Download PhysioNet (MIMIC-IV 3.1, MIMIC-IV-Note 2.2, MIMIC-IV-Ext-CDM 1.1) and the PhysicianBench FHIR image.
# Run on a Bridges-2 LOGIN or DATA-TRANSFER node (compute nodes have ~10-30 KB/s outbound), ideally in tmux:
#   tmux new -s dl 'bash /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor/scripts/download_data.sh'
# Resumable (wget -c); re-run after an interruption. Credentials are read from ENV_FILE and never printed.
#   ONLY=physionet | ONLY=physicianbench   to run one half.
set -uo pipefail
ENV_FILE=${ENV_FILE:-/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor/.env}
DATA=/ocean/projects/cis260032p/yliy/data
PAR=${PAR:-6}                      # parallel connections
REDIVIS_PY=/ocean/projects/cis260032p/yliy/venvs/redivis/bin/python
umask 077
mkdir -p "$DATA/physionet" "$DATA/physicianbench"; chmod 700 "$DATA" "$DATA/physionet" "$DATA/physicianbench"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

if [[ "${ONLY:-physionet}" == physionet || -z "${ONLY:-}" ]]; then
  python3 - "$ENV_FILE" "$TMP/wgetrc" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
u = re.search(r'username:\s*(\S+)', t).group(1); p = re.search(r'password:\s*(\S+)', t).group(1)
open(sys.argv[2], 'w').write(f"user = {u}\npassword = {p}\nauth_no_challenge = on\n")
EOF
  for proj in mimic-iv-note/2.2 mimic-iv-ext-cdm/1.1 mimiciv/3.1; do
    dest="$DATA/physionet/$proj"; mkdir -p "$dest"
    echo "=== $proj $(date +%T)"
    wget --config="$TMP/wgetrc" -q -O "$dest/SHA256SUMS.txt" "https://physionet.org/files/$proj/SHA256SUMS.txt" || { echo "cannot fetch file list for $proj"; continue; }
    awk '{print $2}' "$dest/SHA256SUMS.txt" | xargs -P "$PAR" -I{} sh -c \
      'mkdir -p "$(dirname "$1/$2")" && wget --config="$0" -q -c -O "$1/$2" "https://physionet.org/files/$3/$2" || echo "FAILED $2"' \
      "$TMP/wgetrc" "$dest" {} "$proj"
    (cd "$dest" && sha256sum -c --quiet SHA256SUMS.txt && echo "checksums OK: $proj") || echo "CHECKSUM PROBLEMS: $proj (re-run to resume)"
    du -sh "$dest"
  done
fi

if [[ "${ONLY:-physicianbench}" == physicianbench || -z "${ONLY:-}" ]]; then
  echo "=== PhysicianBench (Redivis) $(date +%T)"
  "$REDIVIS_PY" -I - "$ENV_FILE" "$DATA/physicianbench" <<'EOF'
import os, re, sys
for l in open(sys.argv[1]):
    m = re.match(r'^\s*REDIVIS_API_KEY\s*=\s*(\S+)', l)
    if m: os.environ['REDIVIS_API_TOKEN'] = m.group(1).strip('"\'')
import redivis
table = redivis.organization("AIMI").dataset("physicianbench:a0ek").table("physicianbench:k2n9")
for f in table.list_files():
    print("downloading", f.name)
    f.download(sys.argv[2], overwrite=False)
print("done")
EOF
  ls -la "$DATA/physicianbench"
fi
echo "finished $(date +%T)"
