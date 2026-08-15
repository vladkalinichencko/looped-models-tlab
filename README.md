# Looped transformer на FineWeb

Тестовое задание T-LAB, Looped Models. Условие — [NOTES.md](NOTES.md),
конвенции репозитория — [AGENTS.md](AGENTS.md).

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Train

```bash
python train.py --tag smoke --tokens 200000 --eval-every 50    # smoke-тест
python train.py --tag loop4 --n-loops 4 --tokens 100000000
```

Артефакты — `runs/<tag>/ckpt.pt` и `runs/<tag>/history.json`.

## Eval

```bash
python eval.py runs/loop4/ckpt.pt --loops 1 2 4 8 16 32
```
