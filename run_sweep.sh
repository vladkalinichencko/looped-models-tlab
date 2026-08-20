#!/usr/bin/env bash
# Базовая кривая задания: одни и те же параметры и токены, разное число лупов.
#
# Бюджет 25M токенов вместо разрешённых 100M — на M1 Max один прогон с 16 лупами
# стоит 4.6 часа при 25M и 18 часов при 100M (tmp/bench.txt). Бюджет одинаков у
# всех вариантов, поэтому сравнение честное; финальную конфигурацию гоняем на 100M.
#
#   ./run_sweep.sh                 # свип по числу лупов
#   TOKENS=50000000 ./run_sweep.sh # тот же свип на большем бюджете
set -euo pipefail
cd "$(dirname "$0")"

TOKENS="${TOKENS:-25000000}"

for n in 1 2 4 8 16; do
  .venv/bin/python -u train.py --tag "loops${n}" --n-loops "$n" --tokens "$TOKENS" \
    --diag-every 250 --eval-every 250
done
