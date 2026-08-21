"""Сколько стоит один луп на этой машине — чтобы выбрать бюджет токенов, а не гадать."""
import sys, time, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import Config, LoopedLM

dev = sys.argv[1] if len(sys.argv) > 1 else "mps"
for L in (1, 2, 4, 8, 16):
    m = LoopedLM(Config(vocab_size=16384, n_loops=L)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, 16384, (16, 512), device=dev)
    y = torch.randint(0, 16384, (16, 512), device=dev)
    for _ in range(3):
        _, l = m(x, y); l.backward(); opt.step(); opt.zero_grad()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(5):
        _, l = m(x, y); l.backward(); opt.step(); opt.zero_grad()
    torch.mps.synchronize()
    tps = 5 * 16 * 512 / (time.time() - t0)
    print(f"loops={L:>3}  {tps/1e3:7.1f}k tok/s   25M токенов = {25e6/tps/3600:5.2f} ч   "
          f"100M = {100e6/tps/3600:5.2f} ч", flush=True)
    del m, opt
    torch.mps.empty_cache()
