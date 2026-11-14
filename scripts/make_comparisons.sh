#!/bin/bash
# Makes ALL comparison pictures for the methods evaluated so far, for the three mask types.
#   Style 1: masked input | one method's result            (one picture per method per mask)
#   Style 2: masked input | every method | original         (one picture per mask)
# Uses the 48-image runs, so every method shows the SAME faces. Run from the project folder, environment active:
#   bash scripts/make_comparisons.sh            (8 faces per picture; ROWS=12 bash scripts/make_comparisons.sh for more)
# Needs the MAE 48-image run: python scripts/m1_mae.py eval --mask <m> --n 48 --image-size 128 --name mae_n48 --save-dir outputs/samples/mae_n48_<m>
set -e
cd "$(dirname "$0")/.."
ROWS="${ROWS:-8}"; N="${N:-48}"
METHODS="mae_n48 repaint_uvit mirror_fixed_uvit mirror_adaptive_uvit"
for MASK in one_sided centre square; do
  echo "=== $MASK: style 1 (masked | one method) ==="
  for M in $METHODS; do
    python scripts/make_grid.py --mask $MASK --methods $M --n $N --rows $ROWS --no-original
  done
  echo "=== $MASK: style 2 (all methods side by side) ==="
  python scripts/make_grid.py --mask $MASK --methods $METHODS --n $N --rows $ROWS
done
echo "Done. Pictures are in outputs/ (grid_<mask>__...png)."
ls outputs/grid_*
