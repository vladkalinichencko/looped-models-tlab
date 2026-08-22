#!/usr/bin/env bash
# Отсев идей: каждая — один короткий прогон при одинаковом бюджете и одном сиде.
#
# Смысл не в том, чтобы получить финальные числа, а в том, чтобы понять, шевелится ли
# идея вообще. Числа для сдачи считаются на полном бюджете на A100.
#
# Бюджет 8M токенов при четырёх лупах — примерно полчаса на прогон. У всех идей он
# одинаковый, включая базовую строчку, иначе сравнивать нечего.
#
#   ./run_ideas.sh              # все
#   ./run_ideas.sh deep ponder  # только названные
set -euo pipefail
cd "$(dirname "$0")"

TOKENS="${TOKENS:-8000000}"
LOOPS="${LOOPS:-4}"
COMMON=(--tokens "$TOKENS" --n-loops "$LOOPS" --grad-checkpoint --diag-every 100 --eval-every 100)

SELECT=("$@")
[ ${#SELECT[@]} -eq 0 ] && SELECT=(base layer group inject add step norm deep ponder ouro progress
                                  huginn prelude)
has () { for w in "${SELECT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

run () {
  local tag="$1"; shift
  # по результату, а не по наличию файла: оборванный прогон оставляет частичный
  # history.json, и следующий запуск молча засчитывает его за готовый
  grep -q "best val ppl" "tmp/idea_${tag}.log" 2>/dev/null && { echo "=== idea_${tag}: уже есть"; return; }
  echo "=== idea_${tag}"
  .venv/bin/python -u train.py --tag "idea_${tag}" "${COMMON[@]}" "$@" \
    > "tmp/idea_${tag}.log" 2>&1
  echo "  $(grep -o 'best val ppl [0-9.]*' "tmp/idea_${tag}.log" || echo 'не досчитал')"
}

has base     && run base
has layer    && run layer    --loop-scheme layer
has group    && run group    --loop-scheme group
has inject   && run inject   --input-injection concat
has add      && run add      --input-injection add
has step     && run step     --step-cond
has norm     && run norm     --loop-norm
has deep     && run deep     --deep-supervision 0.3
has ponder   && run ponder   --early-exit pondernet
has ouro     && run ouro     --early-exit ouro
has progress && run progress --progress-head
# Huginn целиком: конкатенация с адаптером, лог-нормальное число повторов, усечённый
# бэкпроп через последние k шагов, prelude и coda вне цикла
has huginn   && run huginn   --input-injection concat --loop-sampling lognormal \
  --loop-mean "$LOOPS" --backprop-last 4 --n-layers 2 --n-prelude 1 --n-coda 1
has prelude  && run prelude  --n-layers 2 --n-prelude 1 --n-coda 1
