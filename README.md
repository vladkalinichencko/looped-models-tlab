# Looped transformer на FineWeb

Тестовое задание T-LAB, направление Looped Models. Условие — в [NOTES.md](NOTES.md),
итоговый отчёт со всеми экспериментами и отрицательными результатами — в
[report.md](report.md).

## Лучший чекпойнт

**https://huggingface.co/vladotpad/looped-qwen3-huginn-fineweb**

Huginn с 16 повторами, 9 147 136 non-embedding параметров, selection perplexity 115.65
на token-matched срезе в 24 584 192 токена. Обычный Qwen3 в один проход на том же срезе
даёт 128.48. Токенизатор лежит там же, в `tokenizer/`.

## Результаты

| модель | повторов | применений блока | selection ppl |
|---|---:|---:|---:|
| Qwen3, один проход | 1 | 4 | 128.48 |
| Huginn | 16 | 32 | 115.65 |
| Антисимметричный переход | 16 | 64 | 370.70 |

Один сид, bf16, A100. Подробности и разбор того, почему антисимметричный вариант
проиграл, — в [report.md](report.md).

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data.py
```

## Обучение

```bash
python train.py
```

Обучает финальный вариант отчёта, Huginn с 16 повторами, на бюджете задания.
Конфигурации задаются Python-объектами внутри файла, CLI-флагов обучения нет. Код
оркестрации прогонов на кластере вынесен в отдельный репозиторий и здесь не лежит.
Каждый прогон пишет `config.json`, `metrics.jsonl`, `history.json`, `best.pt`,
`last.pt`, snapshots held-out тензоров и самодостаточный `report.html`.

## Оценка чекпойнта

```bash
python eval.py runs/huginn-clean-mac/best.pt
```

## Раскладка кода

| путь | что там |
|---|---|
| `model.py` | Qwen3-блоки и `LoopedLM` с явными `encode` / `states` / `decode` |
| `methods/` | по файлу на схему лупинга: `plain`, `huginn`, `antisymmetric`, `controller` |
| `data.py` | закреплённый split FineWeb, чистый BPE, общие token blocks |
| `train.py` | один train- и selection-eval path на все варианты |
| `eval.py` | оценка чекпойнта на фиксированных глубинах |
| `diag.py`, `viz.py` | состояния, логиты, градиенты и самодостаточный HTML из них |
| `make_figures.py` | рисунок отчёта из истории A100 |

Интерактивная диагностика всех прогонов, включая финальные A100, —
[report.html](report.html).
