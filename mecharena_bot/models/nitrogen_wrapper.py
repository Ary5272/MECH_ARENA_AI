"""
CPU-aware wrapper around the NitroGen inference stack.

Key differences from the upstream serve.py / InferenceSession:
- Works on CPU (no hard-coded .to("cuda"))
- Loads the model with strict=False so the checkpoint's game_embedding weights
  are silently ignored when we run in unconditional (game=None) mode.
- No ZMQ server; everything runs in-process.
- predict() returns numpy arrays ready for the action mapper.
"""

import sys
import json
from collections import deque

import torch
import numpy as np

import config

# Make the cloned nitrogen package importable
if config.NITROGEN_PATH not in sys.path:
    sys.path.insert(0, config.NITROGEN_PATH)

from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig, NitrogenTokenizer
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from transformers import AutoImageProcessor


def _load_checkpoint(checkpoint_path: str, device: str):
    """
    Load ng.pt and return (model, tokenizer, img_proc, ckpt_config).

    Game conditioning is intentionally disabled: the checkpoint was trained with
    a game-mapping that references private parquet files.  We set game_mapping=None
    which puts the model in unconditional mode (game_embedding padding_idx=0 → zero
    vector).  The game_embedding weights from the checkpoint are skipped via
    strict=False; all other weights are loaded normally.
    """
    print(f"[NitroGen] Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ckpt_config: CkptConfig = CkptConfig.model_validate(checkpoint["ckpt_config"])
    model_cfg: NitroGen_Config = ckpt_config.model_cfg
    tokenizer_cfg: NitrogenTokenizerConfig = ckpt_config.tokenizer_cfg

    print(f"[NitroGen] Vision encoder : {model_cfg.vision_encoder_name}")
    print(f"[NitroGen] Action horizon : {model_cfg.action_horizon}")
    print(f"[NitroGen] Action dim     : {model_cfg.action_dim}")

    # Disable game mapping — parquet files aren't available locally
    tokenizer_cfg.game_mapping_cfg = None
    tokenizer_cfg.training = False

    print("[NitroGen] Downloading / loading SigLIP image processor ...")
    img_proc = AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)

    print("[NitroGen] Building tokenizer (unconditional mode) ...")
    tokenizer = NitrogenTokenizer(tokenizer_cfg)  # game_mapping will be None

    print("[NitroGen] Building model architecture (downloads SigLIP weights on first run) ...")
    model = NitroGen(config=model_cfg, game_mapping=None)

    print("[NitroGen] Loading checkpoint weights (strict=False, skipping game_embedding) ...")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing:
        print(f"[NitroGen] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[NitroGen] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.eval()
    model.to(device)
    print(f"[NitroGen] Model ready on {device}.")

    return model, tokenizer, img_proc, ckpt_config


class NitroGenWrapper:
    """
    Wraps NitroGen for single-frame inference.

    Usage
    -----
    wrapper = NitroGenWrapper(config.MODEL_PATH)
    # pass a 256×256 BGR numpy frame
    result = wrapper.predict(frame)
    # result = {"j_left": (16,2), "j_right": (16,2), "buttons": (16,21)}
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or config.DEVICE
        self.model, self.tokenizer, self.img_proc, self.ckpt_config = \
            _load_checkpoint(checkpoint_path, self.device)

        modality = self.ckpt_config.modality_cfg
        self.max_buffer = modality.frame_per_sample
        self.obs_buffer: deque = deque(maxlen=self.max_buffer)

    # ------------------------------------------------------------------
    def reset(self):
        self.obs_buffer.clear()

    # ------------------------------------------------------------------
    def predict(self, frame_bgr: np.ndarray) -> dict:
        """
        Args
        ----
        frame_bgr : np.ndarray  shape (H, W, 3), uint8, BGR colour order

        Returns
        -------
        dict with keys:
            j_left   : np.ndarray (action_horizon, 2)  values in [-1, 1]
            j_right  : np.ndarray (action_horizon, 2)  values in [-1, 1]
            buttons  : np.ndarray (action_horizon, 21) values in {0, 1}
        """
        # BGR → RGB (AutoImageProcessor expects RGB)
        frame_rgb = frame_bgr[:, :, ::-1].copy()

        # Encode with SigLIP image processor
        pixel_values = self.img_proc([frame_rgb], return_tensors="pt")["pixel_values"]
        self.obs_buffer.append(pixel_values)

        available = len(self.obs_buffer)
        stacked = torch.cat(list(self.obs_buffer), dim=0)

        # Pad to max_buffer with zero frames; mark padded frames as dropped
        frames = torch.zeros(
            (self.max_buffer, *stacked.shape[1:]),
            dtype=stacked.dtype,
        )
        frames[-available:] = stacked

        dropped = torch.zeros(self.max_buffer, dtype=torch.bool)
        dropped[: self.max_buffer - available] = True

        data = {"frames": frames, "dropped_frames": dropped, "game": None}
        tokenized = self.tokenizer.encode(data)

        # Batch dimension + move to device
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, np.ndarray):
                tokenized[k] = torch.tensor(v, device=self.device).unsqueeze(0)
            else:
                tokenized[k] = [v]

        with torch.inference_mode():
            model_output = self.model.get_action(tokenized, old_layout=False)
            predicted = self.tokenizer.decode(model_output)

        # squeeze batch dim → (action_horizon, dim)
        j_left  = predicted["j_left"].squeeze(0).cpu().numpy()   # (16, 2)
        j_right = predicted["j_right"].squeeze(0).cpu().numpy()  # (16, 2)
        buttons = predicted["buttons"].squeeze(0).cpu().numpy()  # (16, 21)

        return {"j_left": j_left, "j_right": j_right, "buttons": buttons}
