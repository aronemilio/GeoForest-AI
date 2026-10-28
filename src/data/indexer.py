"""Indexación de chips en JSON de metadatos.

Uso:
    python -m src.data.indexer --split train   # -> data/processed/train_index.json
    python -m src.data.indexer --split val     # -> data/processed/val_index.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import rasterio

log = logging.getLogger(__name__)

INDEX_KEYS = ("patch_id", "t0_path", "t1_path", "mask_path", "has_change", "positive_pixels")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_index(split_dir: str | Path, out_json: str | Path, project_root: str | Path = ".") -> list[dict]:
    split_dir, out_json, root = Path(split_dir), Path(out_json), Path(project_root)
    mask_dir = split_dir / "mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"No existe {mask_dir}")

    records = []
    for mask_path in sorted(mask_dir.glob("patch_*.tif")):
        t0 = split_dir / "t0" / mask_path.name
        t1 = split_dir / "t1" / mask_path.name
        missing = [p for p in (t0, t1) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Chip incompleto {mask_path.stem}: faltan {missing}")
        with rasterio.open(mask_path) as m:
            positives = int((m.read(1) > 0).sum())
        records.append(
            {
                "patch_id": mask_path.stem,
                "t0_path": _rel(t0, root),
                "t1_path": _rel(t1, root),
                "mask_path": _rel(mask_path, root),
                "has_change": int(positives > 0),
                "positive_pixels": positives,
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    n_pos = sum(r["has_change"] for r in records)
    log.info("Índice %s: %d chips (%d con cambio)", out_json, len(records), n_pos)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--patches-dir", default="data/processed/patches")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index(
        Path(args.patches_dir) / args.split,
        Path(args.out_dir) / f"{args.split}_index.json",
        args.project_root,
    )


if __name__ == "__main__":
    main()
