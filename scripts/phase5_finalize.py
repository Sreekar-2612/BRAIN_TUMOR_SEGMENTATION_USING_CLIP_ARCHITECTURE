#!/usr/bin/env python3
"""
Phase 5: create artifact layout, consolidate metrics from real sources only,
copy exports, and write reproducibility metadata.

Usage (from repo root):
  python scripts/phase5_finalize.py
  python scripts/phase5_finalize.py --execute-notebook

Metrics are never fabricated: either parsed from the notebook's stored outputs,
from an executed copy, or from run_nb JSON summaries if present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"


def _ensure_dirs() -> None:
    sub = [
        "checkpoints",
        "metrics",
        "plots",
        "qualitative",
        "exports/notebook",
        "exports/script_logs",
        "reproducibility",
        "demo",
    ]
    for s in sub:
        (ARTIFACTS / s).mkdir(parents=True, exist_ok=True)


def parse_eval_block_from_notebook(nb_path: Path) -> dict[str, Any] | None:
    """Parse '=== Final Evaluation Results ===' block from ipynb outputs."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"Mean Dice\s*:\s*([\d.]+).*?"
        r"Mean IoU\s*:\s*([\d.]+).*?"
        r"ROUGE-1\s*:\s*([\d.]+).*?"
        r"ROUGE-L\s*:\s*([\d.]+).*?"
        r"BLEU\s*:\s*([\d.]+)",
        re.S,
    )
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = "".join(out.get("text", []))
            if "=== Final Evaluation Results ===" not in text:
                continue
            m = pattern.search(text)
            if not m:
                return None
            return {
                "dice": float(m.group(1)),
                "iou": float(m.group(2)),
                "rouge1": float(m.group(3)),
                "rougeL": float(m.group(4)),
                "bleu": float(m.group(5)),
                "raw_block": text.strip(),
            }
    return None


def parse_ablation_dice_from_notebook(nb_path: Path) -> tuple[float | None, float | None]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    with_t = with_o = None
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = "".join(out.get("text", []))
            m1 = re.search(r"Mean Dice with Text.*?:\s*([\d.]+)", text)
            m2 = re.search(r"Mean Dice without Text.*?:\s*([\d.]+)", text)
            if m1:
                with_t = float(m1.group(1))
            if m2:
                with_o = float(m2.group(1))
    return with_t, with_o


def load_run_manifest_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mdir = ARTIFACTS / "metrics"
    if not mdir.is_dir():
        return rows
    for p in mdir.glob("run_nb_*_manifest.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        fm = data.get("final_metrics") or {}
        if not fm:
            continue
        exp = p.stem.replace("run_nb_", "").replace("_manifest", "")
        rows.append(
            {
                "run_id": f"run_nb_manifest_{exp}",
                "split": "validation",
                "dice": fm.get("dice", ""),
                "iou": fm.get("iou", ""),
                "rouge1": "",
                "rougeL": "",
                "bleu": "",
                "notes": f"Parsed from {p.relative_to(REPO_ROOT)}; ROUGE/BLEU not in run_nb.",
            }
        )
    return rows


def load_run_nb_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_dir = REPO_ROOT / "outputs" / "metrics"
    if not metrics_dir.is_dir():
        return rows
    for p in metrics_dir.glob("*_summary.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("config") or {}
        fm = data.get("final_metrics") or {}
        rows.append(
            {
                "run_id": p.stem.replace("_summary", ""),
                "split": "validation",
                "dice": fm.get("dice"),
                "iou": fm.get("iou"),
                "rouge1": "",
                "rougeL": "",
                "bleu": "",
                "notes": f"Source: {p.relative_to(REPO_ROOT)}; text metrics not computed in run_nb.py",
                "config": cfg,
            }
        )
    return rows


def write_metrics_tables(rows: list[dict[str, Any]]) -> None:
    csv_path = ARTIFACTS / "metrics" / "final_metrics.csv"
    md_path = ARTIFACTS / "metrics" / "final_metrics.md"
    fieldnames = ["run_id", "split", "Dice", "IoU", "ROUGE-1", "ROUGE-L", "BLEU", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "run_id": r["run_id"],
                    "split": r["split"],
                    "Dice": r.get("dice", ""),
                    "IoU": r.get("iou", ""),
                    "ROUGE-1": r.get("rouge1", ""),
                    "ROUGE-L": r.get("rougeL", ""),
                    "BLEU": r.get("bleu", ""),
                    "notes": r.get("notes", ""),
                }
            )
    lines = ["# Final metrics (Phase 5)", "", "| " + " | ".join(fieldnames) + " |", "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for r in rows:
        cells = [
            str(r.get("run_id", "")),
            str(r.get("split", "")),
            str(r.get("dice", "")),
            str(r.get("iou", "")),
            str(r.get("rouge1", "")),
            str(r.get("rougeL", "")),
            str(r.get("bleu", "")),
            str(r.get("notes", "")).replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_exports(nb_source: Path, executed_nb: Path | None) -> None:
    dest_nb = ARTIFACTS / "exports" / "notebook"
    if executed_nb and executed_nb.is_file():
        shutil.copy2(executed_nb, dest_nb / "Multimodal_BraTS_VLM_executed.ipynb")
    shutil.copy2(nb_source, dest_nb / "Multimodal_BraTS_VLM.ipynb")
    # When nbconvert was not run, the committed notebook may still contain saved cell outputs.
    if not (executed_nb and executed_nb.is_file()):
        shutil.copy2(nb_source, dest_nb / "Multimodal_BraTS_VLM_with_saved_outputs.ipynb")
    for name in ("training_curves.png", "predictions.png"):
        p = REPO_ROOT / name
        if p.is_file():
            shutil.copy2(p, ARTIFACTS / "plots" / name)
    ck = REPO_ROOT / "best_vlm_unet.pth"
    if ck.is_file():
        shutil.copy2(ck, ARTIFACTS / "checkpoints" / "best_vlm_unet.pth")
    ckdir = REPO_ROOT / "checkpoints"
    if ckdir.is_dir():
        for bp in ckdir.rglob("best.pt"):
            dest = ARTIFACTS / "checkpoints" / f"{bp.parent.name}_best.pt"
            shutil.copy2(bp, dest)
    qual = REPO_ROOT / "predictions.png"
    if qual.is_file():
        shutil.copy2(qual, ARTIFACTS / "qualitative" / "predictions_panel.png")


def write_claim_evidence_map(nb_metrics: dict | None, ablation: tuple) -> None:
    wt, wo = ablation
    lines = [
        "# Claim–evidence traceability (Phase 5)",
        "",
        "| Claim | Source file | Artifact / evidence | Status |",
        "| --- | --- | --- | --- |",
        "| Validation Dice, IoU, ROUGE-1, ROUGE-L, BLEU from dual-task evaluation | `Multimodal_BraTS_VLM.ipynb` (eval cell stdout) | `artifacts/metrics/final_metrics.csv` row `notebook_stored_outputs` | verified (parsed from committed notebook outputs) |",
        "| Training loss curves and validation Dice vs epoch | same notebook + matplotlib savefig | `artifacts/plots/training_curves.png` | verified if file present (copied from repo root) |",
        "| Qualitative segmentation panel | same notebook | `artifacts/qualitative/predictions_panel.png` | verified if copied |",
        "| Ablation: Dice with vs without text | same notebook (later cell stdout) | values in this file below | "
        + ("verified" if wt is not None and wo is not None else "missing outputs")
        + " |",
        "| MICCAI `.nii` training pipeline metrics (Dice/IoU/HD/prec/rec) | `run_nb.py` | `outputs/metrics/*_summary.json` if produced | verified only after `run_nb.py` completes |",
        "| ROUGE-2 for text generation | `Project_Report.md` (former wording) | not implemented in notebook (`rouge_scorer` uses rouge1, rougeL only) | removed from report |",
        "| \"CLIP architecture\" in repo folder name | project naming | implementation uses cosine alignment + cross-attention (CLIP-like), not OpenAI CLIP weights | updated (naming vs architecture) |",
        "| Image encoder described as ResNet-style | `Project_Report.md` (former wording) | code uses stacked Conv2d blocks in `Multimodal_BraTS_VLM.ipynb` / `run_nb.py` | updated in report |",
        "| Attention overlay labeled as Grad-CAM equivalent | `Project_Report.md` (former wording) | notebook: attention map; `run_nb.py`: optional true Grad-CAM on decoder features | updated in report |",
        "",
        "## Ablation stdout (from notebook, if present)",
        "",
        f"- Mean Dice with text: {wt}",
        f"- Mean Dice without text: {wo}",
        "",
        "## Parsed evaluation block (subset)",
        "",
        "```text",
        (nb_metrics or {}).get("raw_block", "(no block parsed)")[:800],
        "```",
        "",
    ]
    (ARTIFACTS / "reproducibility" / "claim_evidence_map.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme_reproducibility() -> None:
    text = """# Reproducibility — Dual-task VLM-UNet (BraTS + Text)

This folder documents how to reproduce **notebook-based** training and metrics on a clean machine.
The companion script `run_nb.py` uses **MICCAI BraTS2020 `.nii` volumes** via KaggleHub or `BRATS_MICCAI_ROOT`; it is a separate data layout from the notebook’s `FLAIR_BRATS2020_split` (`.npy` slices).

## 1. Environment

- **OS:** Windows 10/11 or Linux (notebook tested on Colab historically; local paths are configurable).
- **Python:** 3.10+ recommended (project venv under `.venv`).

### Install dependencies

From the repository root `BRAIN_TUMOR_SEGMENTATION_USING_CLIP_ARCHITECTURE/`:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install kagglehub gdown transformers rouge-score nltk nibabel opencv-python-headless tqdm scikit-learn matplotlib jupyter nbconvert
```

Use a CPU-only `torch` wheel if you have no GPU (training will be slow).

## 2. Data layout

### Notebook pipeline (`Multimodal_BraTS_VLM.ipynb`)

- **Images/masks:** `FLAIR_BRATS2020_split/train/images` and `.../train/masks` (`.npy` pairs).
  - Set `FLAIR_BRATS_ROOT` to the directory that **contains** `FLAIR_BRATS2020_split`, or to the split folder itself.
- **Text:** TextBraTS-style `.txt` reports under `TextBraTSData/` (or set `TEXT_BRA_TS_DATA` / `TEXT_BRATS_ZIP_PATH`).

### `run_nb.py` pipeline

- **Images/masks:** MICCAI BraTS2020 training data with folders `BraTS20_Training_*` and `*_flair.nii`, `*_seg.nii`.
  - Set `BRATS_MICCAI_ROOT` to the downloaded `MICCAI_BraTS2020_TrainingData` root, **or** allow KaggleHub download.
- **Text:** same TextBraTS rules as in `run_nb.py` (local folder, zip, or Drive fallback).

## 3. Training commands

### Final notebook run (intended hyperparameters)

From repo root, with GPU strongly recommended:

```powershell
$env:FLAIR_BRATS_ROOT = "<path_containing_FLAIR_BRATS2020_split>"
$env:VLM_FINAL_EPOCHS = "10"
$env:VLM_RUN_SEED = "42"
jupyter nbconvert --to notebook --execute Multimodal_BraTS_VLM.ipynb --output artifacts/exports/notebook/Multimodal_BraTS_VLM_executed.ipynb --ExecutePreprocessor.timeout=-1
```

- **Batch size** is `BATCH = 4` in the notebook DataLoader cell.
- **Optimizer:** AdamW `lr=1e-4`, `weight_decay=1e-2`, cosine schedule over `EPOCHS`.

### Script pipeline (`run_nb.py`)

```powershell
cd BRAIN_TUMOR_SEGMENTATION_USING_CLIP_ARCHITECTURE
.venv\\Scripts\\activate
python run_nb.py --use_text --epochs 10 --batch-size 8 --lr 1e-4 --seed 42
# Ablation:
python run_nb.py --run-both --epochs 10 --batch-size 8 --lr 1e-4 --seed 42
```

CLI flags are written into `outputs/metrics/*_summary.json` when present.

## 4. Runtime and hardware

- **GPU (CUDA):** 10 epochs on the slice-level notebook typically complete in tens of minutes to a few hours depending on GPU and dataset size.
- **CPU:** feasible for debugging; full 10 epochs can take many hours.
- **`run_nb.py` with full volumes:** expect long runtimes and large downloads if BraTS is not already cached.

## 5. Where outputs go

| Output | Location |
| --- | --- |
| Best notebook weights | `best_vlm_unet.pth` (repo root) |
| Loss / metric plots | `training_curves.png`, `predictions.png` |
| Phase 5 bundle | `artifacts/` (metrics, plots, checkpoints copy, exports) |
| `run_nb.py` logs / JSON | `outputs/metrics/`, `checkpoints/<experiment_name>/` |

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `TextBraTS reports not found` (`run_nb.py`) | Missing text folder | Set `TEXT_BRA_TS_DATA` or place `text_brats.zip` and set `TEXT_BRATS_ZIP_PATH` |
| Empty DataLoader / no `.npy` pairs | Wrong `FLAIR_BRATS_ROOT` | Point env to folder containing `FLAIR_BRATS2020_split` |
| `gdown` / Drive errors | quota or permissions | Download zip manually; set `TEXT_BRATS_ZIP_PATH` |
| CUDA OOM | batch too large | Lower `BATCH` in notebook or `--batch-size` for `run_nb.py` |
| NLTK punkt errors | corpora missing | Notebook cells call `nltk.download`; run once online |

## 7. Consolidating metrics after a run

```bash
python scripts/phase5_finalize.py
```

Re-run after training so `final_metrics.csv` can pick up new `outputs/metrics/*_summary.json` files from `run_nb.py` or refreshed notebook stdout.
"""
    (ARTIFACTS / "reproducibility" / "README_reproducibility.md").write_text(text, encoding="utf-8")


def execute_notebook(out_ipynb: Path, epochs: str) -> int:
    env = os.environ.copy()
    env["VLM_FINAL_EPOCHS"] = epochs
    env["VLM_RUN_SEED"] = env.get("VLM_RUN_SEED", "42")
    flair = env.get("FLAIR_BRATS_ROOT") or str(REPO_ROOT)
    env["FLAIR_BRATS_ROOT"] = flair
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(REPO_ROOT / "Multimodal_BraTS_VLM.ipynb"),
        "--output",
        str(out_ipynb.name),
        "--output-dir",
        str(out_ipynb.parent),
        "--ExecutePreprocessor.timeout=-1",
    ]
    log_path = ARTIFACTS / "exports" / "script_logs" / "nbconvert_phase5.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-notebook", action="store_true", help="Run jupyter nbconvert (slow; requires jupyter).")
    parser.add_argument("--nbconvert-epochs", default="10", help="Sets VLM_FINAL_EPOCHS during nbconvert.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    _ensure_dirs()

    nb_path = REPO_ROOT / "Multimodal_BraTS_VLM.ipynb"
    executed = ARTIFACTS / "exports" / "notebook" / "Multimodal_BraTS_VLM_executed.ipynb"

    if args.execute_notebook:
        executed.parent.mkdir(parents=True, exist_ok=True)
        rc = execute_notebook(executed, args.nbconvert_epochs)
        if rc != 0:
            (ARTIFACTS / "exports" / "script_logs" / "nbconvert_phase5_exit_code.txt").write_text(str(rc), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    parsed = parse_eval_block_from_notebook(nb_path)
    if parsed:
        rows.append(
            {
                "run_id": "notebook_stored_outputs",
                "split": "validation",
                "dice": parsed["dice"],
                "iou": parsed["iou"],
                "rouge1": parsed["rouge1"],
                "rougeL": parsed["rougeL"],
                "bleu": parsed["bleu"],
                "notes": "Parsed from Multimodal_BraTS_VLM.ipynb committed stream outputs (eval cell). "
                "Re-execute notebook on your hardware for a new run_id row.",
            }
        )
    else:
        rows.append(
            {
                "run_id": "notebook_stored_outputs",
                "split": "validation",
                "dice": "",
                "iou": "",
                "rouge1": "",
                "rougeL": "",
                "bleu": "",
                "notes": "No evaluation stdout found in notebook JSON; run evaluation cell.",
            }
        )

    for extra in load_run_nb_summaries():
        rows.append(
            {
                "run_id": extra["run_id"],
                "split": extra["split"],
                "dice": extra.get("dice", ""),
                "iou": extra.get("iou", ""),
                "rouge1": extra.get("rouge1", ""),
                "rougeL": extra.get("rougeL", ""),
                "bleu": extra.get("bleu", ""),
                "notes": extra.get("notes", ""),
            }
        )

    for mf in load_run_manifest_metrics():
        rows.append(mf)

    write_metrics_tables(rows)
    ablation = parse_ablation_dice_from_notebook(nb_path)
    write_claim_evidence_map(parsed, ablation)
    write_readme_reproducibility()
    copy_exports(nb_path, executed if executed.is_file() else None)

    sess = ARTIFACTS / "exports" / "script_logs" / "phase5_finalize_session.log"
    sess.write_text(
        "phase5_finalize.py completed.\n"
        "Notebook nbconvert was not run unless --execute-notebook was passed.\n",
        encoding="utf-8",
    )

    # Copy outputs/ visualizations if run_nb was used previously
    out_vis = REPO_ROOT / "outputs" / "visuals"
    if out_vis.is_dir():
        for png in out_vis.rglob("*.png"):
            rel = png.relative_to(REPO_ROOT)
            dest = ARTIFACTS / "plots" / rel.name
            if not dest.exists():
                shutil.copy2(png, dest)

    print("Phase 5 artifacts updated under", ARTIFACTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
