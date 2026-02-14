from __future__ import annotations

"""
Smoke test for the DTI pipeline.

Run (after installing dependencies):
    python examples/smoke_test.py
"""

from pathlib import Path

import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "examples" / "tiny_dti.csv"
    outdir = root / "runs" / "smoke_test_logreg"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(root / "train.py"),
        "--data",
        str(data),
        "--outdir",
        str(outdir),
        "--model",
        "logreg",
        "--cv",
        "2",
        "--drop_invalid",
        "--save_roc",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    print("Saved:", outdir / "model.joblib")
    print("Saved:", outdir / "report.json")


if __name__ == "__main__":
    main()

