from .antisymmetric import AntisymmetricLoopedLM
from .controller import ControllerLoopedLM
from .huginn import HuginnLoopedLM
from .plain import PlainLoopedLM

MODELS = {
    "baseline": PlainLoopedLM,
    "huginn": HuginnLoopedLM,
    "antisymmetric": AntisymmetricLoopedLM,
    "controller": ControllerLoopedLM,
}

def build(cfg):
    return MODELS[cfg.method](cfg)
