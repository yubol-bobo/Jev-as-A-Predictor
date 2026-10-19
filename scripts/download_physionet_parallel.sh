#!/bin/bash
# Faster PhysioNet download: PhysioNet throttles each connection (about 20-30 KB/s), so split every file into
# 32 MB byte ranges and fetch PAR ranges at once. PhysioNet also caps concurrent connections per user: 16 were
# refused (short error responses), 3-4 worked, so PAR defaults to 4. Run on a LOGIN or DATA-TRANSFER node in tmux, after stopping
# scripts/download_data.sh (both write the same files):
#   tmux new -s dlp 'bash /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor/scripts/download_physionet_parallel.sh'
# Resumable: finished files (size and sha256 OK) are skipped; a partial file from download_data.sh is kept as the
# prefix; finished chunks are kept; re-run after any interruption. Credentials come from ENV_FILE and are never
# printed. PROJECTS can restrict the list, e.g. PROJECTS="mimiciv/3.1".
set -uo pipefail
ENV_FILE=${ENV_FILE:-/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor/.env}
DATA=/ocean/projects/cis260032p/yliy/data/physionet
PAR=${PAR:-4}
CHUNK=$((32 * 1024 * 1024))
PROJECTS=${PROJECTS:-"mimic-iv-note/2.2 mimic-iv-ext-cdm/1.1 mimiciv/3.1"}
umask 077
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 - "$ENV_FILE" "$TMP/wgetrc" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
u = re.search(r'username:\s*(\S+)', t).group(1); p = re.search(r'password:\s*(\S+)', t).group(1)
open(sys.argv[2], 'w').write(f"user = {u}\npassword = {p}\nauth_no_challenge = on\ntries = 20\nwaitretry = 10\nread_timeout = 120\n")
EOF
W="wget --config=$TMP/wgetrc"
export W

remote_size() {  # $1 = url. Prints the size only for an HTTP 200 answer; anything else (refused, 429, error page)
  # prints nothing, so a short error body is never mistaken for the file size.
  local h; h=$($W --spider -S "$1" 2>&1 | tr -d '\r')
  grep -qE '^ *HTTP/[0-9.]+ 200' <<<"$h" || return 0
  grep -i 'Content-Length' <<<"$h" | tail -1 | awk '{print $2}'
}

# Range sanity check once: the server must answer 206 with a Content-Range.
probe=$($W -S -O /dev/null --header="Range: bytes=0-1023" "https://physionet.org/files/mimic-iv-note/2.2/LICENSE.txt" 2>&1 | tr -d '\r')
if ! grep -q "206" <<<"$probe" || ! grep -qi "Content-Range" <<<"$probe"; then
  echo "server did not honour Range requests; use scripts/download_data.sh instead"; exit 1
fi
echo "range requests OK; $PAR parallel connections"

: > "$TMP/jobs"
declare -A SIZE
for proj in $PROJECTS; do
  dest="$DATA/$proj"; mkdir -p "$dest"
  [ -s "$dest/SHA256SUMS.txt" ] || $W -q -O "$dest/SHA256SUMS.txt" "https://physionet.org/files/$proj/SHA256SUMS.txt" \
    || { echo "cannot fetch file list for $proj"; continue; }
  while read -r sum rel; do
    f="$dest/$rel"; url="https://physionet.org/files/$proj/$rel"
    mkdir -p "$(dirname "$f")"
    if [ -f "$f.ok" ]; then continue; fi
    size=$(remote_size "$url"); [ -n "$size" ] || { echo "no size for $proj/$rel"; continue; }
    SIZE["$f"]=$size
    have=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 )
    if [ "$have" -eq "$size" ] && echo "$sum  $f" | sha256sum -c --quiet - 2>/dev/null; then touch "$f.ok"; continue; fi
    if [ "$have" -gt "$size" ]; then  # never delete data: report and leave it for a human
      echo "SKIP $proj/$rel: local $have bytes > remote $size; check by hand"; continue
    fi
    # chunks are only valid for one (size, prefix); a stale or unmarked parts dir holds chunks only, so drop it
    if [ -d "$f.parts" ] && [ "$(cat "$f.parts/.layout" 2>/dev/null)" != "$size $have" ]; then rm -rf "$f.parts"; fi
    mkdir -p "$f.parts"; echo "$size $have" > "$f.parts/.layout"
    for ((s = have; s < size; s += CHUNK)); do
      e=$((s + CHUNK - 1)); [ $e -ge $size ] && e=$((size - 1))
      printf '%s %s %s %s\n' "$url" "$f.parts/$(printf '%012d' $s)" "$s" "$e" >> "$TMP/jobs"
    done
    echo "$proj/$rel: $((size / 1048576)) MB, have $((have / 1048576)) MB"
  done < "$dest/SHA256SUMS.txt"
done
echo "$(wc -l < "$TMP/jobs") chunks to fetch, started $(date +%T)"

# fetch chunks; a chunk is done when its size equals e-s+1 (otherwise refetched from scratch)
xargs -P "$PAR" -L 1 sh -c '
  url=$0; out=$1; s=$2; e=$3; want=$((e - s + 1))
  [ -f "$out" ] && [ "$(stat -c %s "$out")" -eq "$want" ] && exit 0
  for i in 1 2 3 4 5; do
    hdr=$($W -S -O "$out.tmp" --header="Range: bytes=$s-$e" "$url" 2>&1 | tr -d "\r")
    echo "$hdr" | grep -qE "^ *HTTP/[0-9.]+ 206" && [ "$(stat -c %s "$out.tmp")" -eq "$want" ] \
      && mv "$out.tmp" "$out" && exit 0
    rm -f "$out.tmp"
    sleep 20
  done
  echo "FAILED chunk $out (re-run to resume)"' < "$TMP/jobs"

# assemble: append chunks in order to the prefix, then verify
for proj in $PROJECTS; do
  dest="$DATA/$proj"
  while read -r sum rel; do
    f="$dest/$rel"; [ -f "$f.ok" ] && continue; [ -d "$f.parts" ] || continue
    size=${SIZE["$f"]:-0}; complete=1; total=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 )
    for c in "$f.parts"/0*; do [ -f "$c" ] && [[ "$c" != *.tmp ]] && total=$((total + $(stat -c %s "$c"))); done
    [ "$total" -eq "$size" ] || { echo "incomplete: $proj/$rel ($((total / 1048576))/$((size / 1048576)) MB)"; continue; }
    for c in $(ls "$f.parts" | grep -E '^[0-9]+$' | sort); do cat "$f.parts/$c" >> "$f"; done
    if echo "$sum  $f" | sha256sum -c --quiet -; then rm -rf "$f.parts"; touch "$f.ok"; echo "OK $proj/$rel"
    else echo "CHECKSUM FAILED $proj/$rel: delete it and its .parts and re-run"; fi
  done < "$dest/SHA256SUMS.txt"
  echo "$proj: $(ls "$dest" -R | grep -c '\.ok$') files verified, $(du -sh "$dest" | cut -f1)"
done
echo "finished $(date +%T)"
