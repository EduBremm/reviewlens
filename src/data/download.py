"""Etapa 1 — Ingestão: baixa o CSV bruto e confere o hash SHA-256.

Uso: python -m src.data.download
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request

from src.config import ROOT, ensure_dirs, params


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ensure_dirs()
    cfg = params()["data"]
    dest = ROOT / cfg["raw_path"]
    if dest.exists() and sha256_of(dest) == cfg["sha256"]:
        print(f"[download] {dest.name} já existe e o hash confere — nada a fazer.")
        return
    print(f"[download] baixando {cfg['url']}")
    urllib.request.urlretrieve(cfg["url"], dest)
    digest = sha256_of(dest)
    if digest != cfg["sha256"]:
        dest.unlink(missing_ok=True)
        sys.exit(f"[download] hash divergente: {digest} != {cfg['sha256']}")
    print(f"[download] ok — {dest.stat().st_size/1e6:.1f} MB, sha256 conferido")


if __name__ == "__main__":
    main()
