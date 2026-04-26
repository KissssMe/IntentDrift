from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from secmcp.config import PROJECT_ROOT
from secmcp.data.io import read_samples_jsonl


def test_unify_script_debug_run(tmp_path):
    script = PROJECT_ROOT / "scripts" / "01_unify_datasets.py"
    out_dir = tmp_path / "splits"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--max-per-source",
            "10",
            "--agentdojo-pipeline",
            "command-r",
            "--output-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "loaded=" in result.stdout
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        assert path.exists()
        rows = read_samples_jsonl(path)
        assert rows
