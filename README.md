# Looped Models

Один recurrent core Qwen3, четыре отдельные модели, как его крутить.

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

`train.py` собирает Huginn напрямую. Остальные методы: `PlainLoopedLM`, `AntisymmetricLoopedLM`, `ControllerLoopedLM`. Чекпойнт поднимается через `LoopedLM(Config(...))` — это только сборка по полю `method` в json.

## Где что лежит

| из отчёта | в коде |
|---|---|
| Qwen3 attention / MLP | `blocks.py` |
| гиперпараметры модели | `config.py` |
| baseline | `methods/plain.py` |
| Huginn | `methods/huginn.py` |
| антисимметричный переход | `methods/antisymmetric.py` |
| контроллер | `methods/controller.py` |
| сборка чекпойнта по `method` | `model.py` |
| данные | `data.py` |
| обучение | `train.py` |
| оценка на фиксированной глубине | `eval.py` |
| диагностика | `diag.py` |
| код для HTML-визуализаций | `viz.py` |
