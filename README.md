# Looped transformer на FineWeb

Тестовое задание T-LAB, Looped Models. Условие — [NOTES.md](NOTES.md),
конвенции репозитория — [AGENTS.md](AGENTS.md).

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Clean Mac preliminary runs

```bash
python -m tmp.test_clean_path
python run_preliminary.py
```

Фиксированные Python-конфигурации находятся в `run_preliminary.py`. Каждый прогон
пишет `config.json`, `metrics.jsonl`, `history.json`, `best.pt`, `last.pt`, snapshots
реальных held-out тензоров и самодостаточный `report.html`.

## Eval

```bash
python eval.py runs/huginn-clean-mac/best.pt
```
