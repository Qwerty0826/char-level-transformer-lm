"""
HuggingFace Spaces entrypoint for the 60M Transformer LM playground.

Downloads the base / SFT / DPO checkpoints + tokenizer files from a
HuggingFace model repo on cold start, then launches the same Gradio
playground that ``scripts/playground.py`` runs locally — including the
Aligned tab that streams the same prompt through all three checkpoints
side by side.

Deploy:
  1. Upload the three checkpoints + tokenizer files to a HF model repo
     (see ``scripts/upload_checkpoints_to_hf.py``).
  2. Create a HuggingFace Space with this directory as its root, set
     hardware to "CPU basic" (or T4 small for paid speed), and set
     ``MODEL_REPO`` in the Space's Variables to your model repo id.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Make the repo root importable so we can reuse scripts/playground.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download

from scripts.playground import build_ui, load_models


MODEL_REPO = os.environ.get("MODEL_REPO", "pragadeeshsk/transformer-lm-60m-tinystories")


def main() -> None:
    print(f"[spaces] Downloading checkpoints from {MODEL_REPO} ...")
    local_dir = snapshot_download(repo_id=MODEL_REPO, repo_type="model")
    local = Path(local_dir)

    args = SimpleNamespace(
        checkpoint=str(local / "base_60m" / "final.pt"),
        checkpoint_sft=str(local / "sft_v2" / "final.pt"),
        checkpoint_dpo=str(local / "dpo_v2" / "final.pt"),
        vocab=str(local / "tinystories_v2_vocab.json"),
        merges=str(local / "tinystories_v2_merges.txt"),
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
        vocab_size=16_000,
        context_length=512,
        d_model=640,
        num_layers=10,
        num_heads=10,
        num_kv_heads=2,
        d_ff=1728,
        theta=10_000.0,
        no_tie_weights=False,
        device=None,        # auto: cuda if available, else cpu
        dtype="float32",    # safe default for Spaces CPU; override on GPU
    )

    models, tokenizer, device, eos_id = load_models(args)
    app = build_ui(models, tokenizer, device, eos_id)
    app.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
