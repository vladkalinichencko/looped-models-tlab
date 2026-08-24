# Looped Models

Зацикленный трансформер на FineWeb: один блок Qwen3 применяется много раз подряд.

Отчёт — [report.md](report.md). Диагностика всех прогонов — [report.html](report.html).
Лучший чекпойнт — https://huggingface.co/vladotpad/looped-qwen3-huginn-fineweb

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data.py                    # кэш FineWeb, токенизатор, блоки токенов
python train.py                   # обучение
python eval.py runs/huginn/best.pt
python viz.py                     # пересборка report.html
```

## Где что лежит

| из отчёта | в коде |
|---|---|
| один проход Qwen3 | `methods/plain.py` |
| Huginn | `methods/huginn.py` |
| антисимметричный переход | `methods/antisymmetric.py` |
| контроллер над слоями | `methods/controller.py` |
| блоки Qwen3 и общий цикл | `model.py` |
| данные и токенизатор | `data.py` |
| обучение и отбор чекпойнта | `train.py` |
| оценка на фиксированных глубинах | `eval.py` |
| состояния, приращения, градиенты | `diag.py` |
| сборка страницы диагностики | `viz.py` |
