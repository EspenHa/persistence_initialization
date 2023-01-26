__all__ = ["GLOBALS"]
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Globals:
    root: Path = Path(__file__).parent.parent
    data_root: Path = root / "data"
    output_root: Path = root / "outputs"


GLOBALS = _Globals()
