from __future__ import annotations

import sys
sys.path.append("")

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR / "UCE-main").resolve()
DEFAULT_MODEL_FILE = "4layer_model.torch"
DEFAULT_NLAYERS = 4

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Project directory not found: {PROJECT_ROOT}")

sys.path.insert(0, str(PROJECT_ROOT))

from model import TransformerModel  # noqa: E402


def to_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def find_first(root: Path, file_name: str) -> Path | None:
    direct = root / file_name
    if direct.exists():
        return direct.resolve()
    for candidate in root.rglob(file_name):
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_model_path(resource_dir: str | None, model_loc: str | None) -> Path:
    if model_loc:
        path = Path(model_loc).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    candidates = [to_path(resource_dir), SCRIPT_DIR, PROJECT_ROOT]
    for root in candidates:
        if root is None or not root.exists():
            continue
        found = find_first(root, DEFAULT_MODEL_FILE)
        if found is not None:
            return found

    raise FileNotFoundError(
        f"Could not find {DEFAULT_MODEL_FILE}. "
        f"Please provide --resource_dir or --model_loc."
    )


def build_model(nlayers: int = DEFAULT_NLAYERS) -> TransformerModel:
    model = TransformerModel(
        token_dim=5120,
        d_model=1280,
        nhead=20,
        d_hid=5120,
        nlayers=nlayers,
        dropout=0.05,
        output_dim=1280,
    )

    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe, freeze=True)
    return model


def normalize_state_dict(state_dict: dict) -> OrderedDict:
    normalized = OrderedDict()
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def load_checkpoint(model_path: Path) -> OrderedDict:
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint content is not a usable state_dict.")

    return normalize_state_dict(checkpoint)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_load_tokens(resource_dir: str | None, token_file: str | None) -> Path | None:
    if token_file:
        path = Path(token_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")
        return path

    root = to_path(resource_dir)
    if root is None or not root.exists():
        return None

    return find_first(root, "all_tokens.torch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load UCE pretrained 4-layer model only, without inference."
    )
    parser.add_argument(
        "--resource_dir",
        type=str,
        default=None,
        help="Directory that contains 4layer_model.torch",
    )
    parser.add_argument(
        "--model_loc",
        type=str,
        default=None,
        help="Explicit path to 4layer_model.torch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Final device for the model. Default: cpu",
    )
    parser.add_argument(
        "--token_file",
        type=str,
        default=None,
        help="Optional explicit path to all_tokens.torch",
    )
    parser.add_argument(
        "--check_tokens",
        action="store_true",
        help="Also try loading all_tokens.torch on CPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.resource_dir, args.model_loc)
    target_device = resolve_device(args.device)

    print("Start loading UCE 4-layer model...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model file: {model_path}")
    print("Checkpoint will be read with map_location='cpu' first.")

    state_dict = load_checkpoint(model_path)
    model = build_model(DEFAULT_NLAYERS)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(target_device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    first_param = next(model.parameters())

    print("Model loaded successfully.")
    print(f"Target device: {target_device}")
    print(f"First parameter device: {first_param.device}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    if args.check_tokens:
        token_path = maybe_load_tokens(args.resource_dir, args.token_file)
        if token_path is None:
            print("all_tokens.torch not found. Token check skipped.")
        else:
            print(f"Checking token file: {token_path}")
            tokens = torch.load(token_path, map_location="cpu")
            shape = tuple(tokens.shape) if hasattr(tokens, "shape") else "unknown"
            print(f"all_tokens.torch loaded successfully, shape={shape}")


if __name__ == "__main__":
    main()
