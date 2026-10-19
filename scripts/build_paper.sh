#!/bin/bash
# Rebuild the paper from the latest results: analyses -> generated tables/numbers -> figures -> PDF (aggregates only).
#   bash scripts/build_paper.sh
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
FIGPY=/ocean/projects/cis260032p/yliy/venvs/figs/bin/python
cd "$REPO"
(cd scripts && ../.venv/bin/python pb_analyze.py --out ../results/pb_analysis.json > /dev/null)
.venv/bin/python scripts/pb_offline_orders.py build --slug clef_k5/run1 > /dev/null && \
  .venv/bin/python scripts/pb_offline_orders.py report --slug clef_k5/run1 --out results/pb_offline_orders_clef_online.json > /dev/null
.venv/bin/python scripts/make_tables.py > /dev/null
$FIGPY paper/figures/src/data_figs.py > /dev/null
[ -f paper/figures/fig1_overview.pdf ] || $FIGPY paper/figures/src/fig1_overview.py > /dev/null
cd paper && for i in 1 2; do pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; [ $i = 1 ] && bibtex main > /dev/null 2>&1; done
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
grep -q "Output written" main.log && echo "paper built: $(grep -o '([0-9]* pages' main.log)"
