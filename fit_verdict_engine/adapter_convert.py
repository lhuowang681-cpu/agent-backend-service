from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence


SWIFT_QWEN35_TARGET_PREFIX = r"model\.language_model(?=\.)"
PEFT_RUNTIME_TARGET_PREFIX = r"model(?=\.)"
SWIFT_QWEN35_STATE_PREFIX = "base_model.model.model.language_model."
PEFT_RUNTIME_STATE_PREFIX = "base_model.model.model."


def convert_adapter_config(config: dict) -> dict:
    converted = dict(config)
    target_modules = converted.get("target_modules")
    if isinstance(target_modules, str):
        converted["target_modules"] = target_modules.replace(
            SWIFT_QWEN35_TARGET_PREFIX,
            PEFT_RUNTIME_TARGET_PREFIX,
        )
    return converted


def convert_state_key(key: str) -> str:
    if key.startswith(SWIFT_QWEN35_STATE_PREFIX):
        return PEFT_RUNTIME_STATE_PREFIX + key[len(SWIFT_QWEN35_STATE_PREFIX) :]
    return key


def _copy_sidecar_files(source_dir: Path, output_dir: Path) -> None:
    for path in source_dir.iterdir():
        if path.name == "adapter_model.safetensors":
            continue
        if path.name == "adapter_config.json":
            continue
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def convert_swift_qwen35_adapter(source_dir: Path, output_dir: Path) -> Path:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("adapter conversion requires safetensors in the active Python environment") from exc

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    config_path = source_dir / "adapter_config.json"
    weights_path = source_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"missing adapter_config.json: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"missing adapter_model.safetensors: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_sidecar_files(source_dir, output_dir)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    converted_config = convert_adapter_config(config)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(converted_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    state = load_file(weights_path)
    converted_state = {convert_state_key(key): value for key, value in state.items()}
    save_file(converted_state, output_dir / "adapter_model.safetensors")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an ms-swift Qwen3.5 LoRA checkpoint for the local PEFT runtime loader."
    )
    parser.add_argument("--source", type=Path, required=True, help="Swift checkpoint directory.")
    parser.add_argument("--output", type=Path, required=True, help="Converted adapter output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = convert_swift_qwen35_adapter(args.source, args.output)
    print(f"Converted adapter written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
