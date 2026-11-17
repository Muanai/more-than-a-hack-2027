# More than a Hack 2027 — Phase 1: Ball Possession (`sendiri aj`)

Frame-level football ball possession segmentation (`H`, `A`, `D`, `X`) across **199,319 frames** (`BAR-ATH`, 25 fps) using physics-based dead-ball/control signals, occlusion-augmented weakly-supervised `H-vs-A` experts, Outer-EM self-training, and an explicit-duration per-half HSMM sequence decoder.

---

## Repository Structure

- `scripts/convert_to_parquet.py` — Converts raw tactical ATD tracking, slot metadata, homography, and Smart Tagging XML into compressed Parquet files under `data/processed/tactical/`.
- `scripts/generate_submission.py` — Deterministic CLI entrypoint producing the submission CSV (`submissions/sendiri aj.csv`).
- `src/data/loader.py` — Loads tactical Parquet data, filters inactive tracking slots (`{22, 23, 44}`), and builds half/graded masks (`11,185..90,651` and `110,808..190,197`).
- `src/features/possession_features.py` — Computes 40 halftime-isolated kinematic, spatial, zero-phase Nadaraya-Watson control, and Smart Tagging features.
- `src/models/weak_emissions.py` — Dropout-augmented weakly-supervised `H-vs-A` ensemble (zero leakage: fit strictly on `graded_mask & (~win_mask)`).
- `src/models/em_refinement.py` — Multi-horizon temporal features, physics-bounded short-stoppage recovery, and Round-2 Outer-EM self-training (`93,436` non-window frames).
- `src/models/hsmm_decoder.py` — Per-half 3-state Viterbi and explicit-duration Block-5 HSMM sequence decoders.
- `src/pipelines/` — Ablation & reproducible end-to-end pipelines (`baseline.py`, `ablation.py`, `step2_learned.py`, `step3_em.py`).
- `src/evaluation/diagnostics.py` — Full-match structural diagnostics (`spells/5min`, `<1s%`, half symmetry, dead-ball share).
- `submissions/sendiri aj.md` — Competition short technical document.

*(Note: Competition data under `data/` and upstream organizer assets under `reference/` are excluded via `.gitignore` and not redistributed.)*

---

## Environment Setup

Python **`3.12.6`** (`>=3.12,<3.13`) with pinned dependencies in `pyproject.toml`, `uv.lock`, and `requirements.txt`.

### Option A — Using `uv` (recommended)

```bash
uv sync --frozen
```

### Option B — Using `pip` + `venv`

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Data Placement & Exact Reproduction

1. Place the four official **tactical** raw files in `data/raw/tactical/`:
   - `data/raw/tactical/BAR-ATH_tactical_atd.txt`
   - `data/raw/tactical/BAR-ATH_tactical_atd.xml`
   - `data/raw/tactical/BAR-ATH_tactical_homography.csv`
   - `data/raw/tactical/BAR-ATH_tactical_smart_tagging.xml`

2. Convert the raw tactical files to Parquet:

   ```bash
   uv run python scripts/convert_to_parquet.py --source tactical
   ```

3. Regenerate the final submission (`submissions/sendiri aj.csv`, deterministic across runs):

   ```bash
   uv run python scripts/generate_submission.py --out "submissions/sendiri aj.csv" --model step3_hsmm
   ```

4. *(Optional)* Verify submission schema using the organizer validator (if `reference/` is present locally):

   ```bash
   uv run python reference/evaluation/validate_submission.py "submissions/sendiri aj.csv"
   ```
