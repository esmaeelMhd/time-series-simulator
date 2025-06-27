from .lstm import SimpleLSTM
from .transformer import SimpleTransformer

MODEL_REGISTRY = {
    "lstm": SimpleLSTM,
    "transformer": SimpleTransformer,
}


def get_model(name: str):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model {name} not found in registry.")
    return MODEL_REGISTRY[key] 