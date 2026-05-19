# Lazy imports so the package can be imported even while individual
# submodules are being developed incrementally.
from __future__ import annotations


def __getattr__(name: str):
    if name in ("Tokenizer", "train_bpe"):
        from cs336_basics.tokenizer import Tokenizer, train_bpe
        return {"Tokenizer": Tokenizer, "train_bpe": train_bpe}[name]
    if name == "TransformerLM":
        from cs336_basics.model import TransformerLM
        return TransformerLM
    if name == "AdamW":
        from cs336_basics.optimizer import AdamW
        return AdamW
    if name in ("cross_entropy_loss", "get_lr_cosine_schedule",
                "clip_gradient_norm", "get_batch",
                "save_checkpoint", "load_checkpoint"):
        import cs336_basics.training as _t
        return getattr(_t, name)
    raise AttributeError(f"module 'cs336_basics' has no attribute {name!r}")
