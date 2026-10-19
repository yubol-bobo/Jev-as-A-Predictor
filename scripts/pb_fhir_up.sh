#!/bin/bash
# Start a fresh PhysicianBench HAPI FHIR server under Apptainer (no Docker on Bridges-2).
# Equivalent of `docker run -p PORT:8080 fhir-full:v1` in PhysicianBench/scripts/run_task.py:
# each run gets its own copy of the pristine H2 database, so writes made by an agent never leak across tasks.
#   bash scripts/pb_fhir_up.sh [PORT] [RUN_DIR]     -> prints PID; base URL http://127.0.0.1:PORT/fhir
# Stop with: kill -- -<PID>  (PID is the process-group leader); rm -rf RUN_DIR
# The image keeps its database at /tmp/hapi-h2db.mv.db. Apptainer would hide that behind the host /tmp,
# so we extract it once (data/physicianbench/pristine/) and bind a writable per-run copy at /tmp.
set -euo pipefail
PORT=${1:-18080}
PB=/ocean/projects/cis260032p/yliy/data/physicianbench
RUN=${2:-${TMPDIR:-/tmp}/pb_fhir_$PORT}
umask 077
mkdir -p "$RUN/tmp"
cp --sparse=always "$PB/pristine/hapi-h2db.mv.db" "$RUN/tmp/"
# Listen on localhost only: compute nodes are shared and this is restricted clinical data.
setsid nohup apptainer run --containall --cleanenv --pwd /app -B "$RUN/tmp:/tmp" "$PB/physicianbench-fhir-v1.sif" \
  --spring.config.location=/configs/application.yaml --server.port="$PORT" --server.address=127.0.0.1 \
  > "$RUN/server.log" 2>&1 &
echo $!
