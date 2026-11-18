import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    """Banco isolado para os testes da API (não suja data/reviewlens.db)."""
    import shutil

    db = tmp_path_factory.mktemp("db") / "test.db"
    src = ROOT / "data" / "reviewlens.db"
    if src.exists():
        shutil.copy(src, db)
    os.environ["REVIEWLENS_DB"] = str(db)
    return db


def requires(path):
    return pytest.mark.skipif(not (ROOT / path).exists(), reason=f"{path} ausente — rode `dvc repro` antes")
