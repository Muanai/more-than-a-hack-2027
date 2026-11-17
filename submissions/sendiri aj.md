# More than a Hack 2027 — Phase 1: Technical Document

**Team:** `sendiri aj`  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`, 199,319 frames @ 25 fps, 158,857 graded frames)  
**Submission:** `sendiri aj.csv` (`ablation8_step3_em_hsmm`)

---

## 1. Approach Overview & Design Rationale

Frame-level possession from uncorrected automatic tracking data (ATD) fails for naive nearest-player baselines for two structural reasons:
1. **Ball dropouts (~40.7% of graded frames missing):** Standard heuristics (`ball_nan -> D`) confuse visual occlusion during live play with stoppages (`40.69%` raw dead-ball prediction full-match vs. `11.80%` in the clean opening window) and shatter continuous spells into `<1s` slivers (`51.77% <1s` raw, `18.09% <1s` even with `±5s` smoothing).
2. **Intra-half tempo decay:** Median active collective player speed falls by **~33%** within a half (`2.083 m/s` at 0–5 min to `1.342 m/s` at 30–45 min) before resetting at halftime, so fixed global speed thresholds misclassify late-half buildup play as dead ball.

To address both without relying on human annotations on `BAR-ATH`, we built a **two-stage decoupled emission + explicit-duration sequence architecture** (`ablation8_step3_em_hsmm`):
- **Dead-ball vs. Alive (`D` vs. `{H, A}`)** is separated cleanly from team ownership (`H` vs. `A`) using tempo-normalized collective player kinematics, causal out-of-bounds tracking, and physics-bounded missing-ball stoppage recovery (`emit_d_enh`), preserving sharp stoppage hysteresis boundaries (`win_alive_dead_F1 = 0.9865`).
- **Team ownership (`H` vs. `A`)** uses two complementary dropout-augmented `0.4 LogisticRegression + 0.6 HistGradientBoosting` experts (`19`-feature occlusion-resilient expert + `30`-feature rich control expert) refined via **Round-2 Outer-EM self-training** strictly outside the public 5-minute benchmark window.
- **Sequence decoding** maps bounded log-odds (`tanh(logit / 3.0)`, smoothed per half over `15` frames) through an **explicit-duration Block-5 Hidden Semi-Markov Model (`0.2s` blocks)** per half (`11,185..90,651` and `110,808..190,197`), eliminating sub-second fragments (`0.0% <1s`) while preserving balanced full-match spell density (`15.77` spells/5min across 105.9 graded minutes; `15.48` H1 vs. `16.06` H2).

We chose regularized linear + GBDT tabular experts over deep temporal sequence networks because they train deterministically in seconds on weak/self-supervised pseudo-labels across the full unlabeled match without overfitting to scarce labels, and pair transparently with dynamic programming constraints (`HSMM` dwell priors and transition costs).

---

## 2. Competition Assets Used, Dropped, and Noise Handling

### Used Assets (`tactical` timeline)
- **Tactical ATD Tracking & Slot Metadata (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`):** Primary positional and speed source (`x, y, speed`, normalized `0..1` -> meters on `105.3m × 68.0m`).
  - *Slot hygiene:* Three dead/phantom slots (`EXCLUDED_SLOT_IDS = {22, 23, 44}`) are removed at load time, leaving 43 active slots.
  - *Attacking direction normalization:* Team coordinates and longitudinal velocities (`all_vx_s`, `h_vx_s`, `a_vx_s`) are flipped in Half 2 so positive `x` always aligns with the same attacking direction.
  - *Ball teleport rejection:* Instantaneous ball displacement jumps corresponding to tracker re-acquisitions (`> 40 m/s`) are masked (`bs_clean`) before velocity/control estimation.
  - *Missing-ball occlusion handling:* Control evidence (`nw_margin`) is weighted smoothly by a ball-control kernel (`w_ctrl`) and propagated exponentially across missing-ball intervals (`carried_nw` at 1s, 2s, and 5s half-lives). Both `H/A` experts are trained with **synthetic dropout augmentation** (randomly masking local ball features and Smart Tagging signals) so occluded intervals fall back cleanly on collective team velocity (`all_vx_s`), line height (`pitch_tilt`), carried memory, and tactical phase cues.
- **Tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`, 638 events):** Used as a non-authoritative soft prior after polarity alignment (`+1` offensive codes, `-1` defensive codes) and deduplication of mirrored opposing defensive/offensive phase pairs (`tag_prior_dedup`) so co-occurring phase tags do not double-count.
- **Tactical Homography (`BAR-ATH_tactical_homography.csv`, 199,319 rows):** Verified at ETL conversion time (`100% valid`); raw ATD normalized coordinates already encode pitch geometry consistently across all tactical frames.

### Tried and Dropped Assets
- **Broadcast Video, ATD & Homography (`164,488` frames):** Dropped early because the broadcast stream sits on an unsynchronized timeline with cuts/replays and only `65.8%` homography validity, offering no clean frame-exact mapping back to the `199,319`-frame tactical reference.
- **Broadcast/Raw `D` Classifier:** Evaluated supervised/weakly-supervised multi-scale `D`-vs-`Alive` classifiers (`LR + HGB` with rolling features) and soft `D` blends; **dropped** because soft regression blurred stoppage onset/exit hysteresis, lowering `alive_dead_F1` (`0.9769 -> 0.9688–0.9756`) relative to the crisp morphological + causal physics detector (`emit_d_enh`, `0.9865`).
- **Joint 3-Class (`H/A/D`) Weak-Anchor Classifier:** Tested and **dropped** because applying missing-ball dropout augmentation to dead-ball and live-ball anchors together caused the classifier to conflate slow live buildup play with `D`, inflating `D` predictions to `34–45%`.

---

## 3. Training Data & Zero-Leakage Protocol

- **External Labelled Datasets:** **None (`0` external games or datasets used).**
- **Human Labels on `BAR-ATH`:** **None (`0` manual labels created, and `0` frames of the 5-minute public ground truth used during training, pseudo-labeling, or Outer-EM fitting).** The public 5-minute window (`11,185..18,684`) is reserved strictly for out-of-sample sanity checks and structural diagnostics.
- **Algorithmic Self-/Weak-Supervision (`src/models/weak_emissions.py`, `src/models/em_refinement.py`):**
  1. **Round 1 High-Precision Anchors (`62,070` frames):** Selected algorithmically on `data.graded_mask & (~data.win_mask)` (`151,357` unlabeled graded frames) where play is live (`~d_mask`) and either local kernel ball control is high (`den_nw >= 0.35`, `|nw_margin| >= 0.32` without tag contradiction) or Smart Tagging polarity is strong (`|tag_prior_dedup| >= 0.75` without control contradiction).
  2. **Round 2 Outer-EM Expansion (`93,436` frames):** Expanded algorithmically on non-window graded frames (`graded_mask & (~win_mask)`) by adding Round-1 HSMM interior frames eroded by `15` frames from segment boundaries where model probability confidence agrees (`>= 0.72`), weighted `0.65` relative to `1.0` for Round-1 anchors.
  3. **Cross-Half Generalization Check:** Training Round-1 experts solely on `H1 \ win` predicts `H2` anchors at **`0.9964`** accuracy (`0.8889` out-of-sample on the 5-min window); training solely on `H2` predicts `H1 \ win` anchors at **`0.9973`** accuracy (`0.8881` out-of-sample on the 5-min window).

---

## 4. Possession Rules Implemented

- **Live Play vs. Stoppage (`D`):**
  - A frame enters candidate dead-ball play via tempo-adjusted slowdown (`slowdown_diff < -0.78 m/s`), causal out-of-bounds triggers (remaining `D` across goal-area resets until ball speed indicates restart `bs_clean >= 5.0 m/s`), or prolonged missing/slow ball intervals, cleaned via morphological opening (`min_true = 115f`) and closing (`min_false = 65f`).
  - Brief missing-ball collective slowdowns (`slowdown_diff < -0.75`, `frac_slow_diff > 0.30`, `bp_s < 0.35` for `>= 35f`, firing on 5 match episodes) are recovered as stoppages and bounded strictly within adjacent `!ball_present & bs_clean < 6.0` frames (`emit_d_enh = +1.60`).
  - Graded pre-kickoff Half 2 frames before actual kickoff (`110,808..113,742`) are captured as `D` by the causal/kinematic pipeline; ungraded frames (`< 11,185`, halftime `90,652..110,807`, `> 190,197`) are emitted deterministically as `X`.
- **Controlled Possession vs. Contests / Deflections (`H` vs. `A`):**
  - Momentary touches during fast ball flight or deflections are suppressed using a speed-and-distance control weight `w_ctrl = exp(-max(0, d_min - 1.5)/2.5) * exp(-max(0, bs_clean - 8.0)/8.0)` inside zero-phase Nadaraya-Watson smoothing (`nw_margin`), so loose balls keep continuous prior team control.
  - Inside the HSMM decoder (`pen_ha = -26.0`, `pen_d = -26.0`, `min_dur_ha = 25f`, `min_dur_d = 45f`, log-normal dwell priors centered at `9s` live / `12s` stoppage), a team switch requires sustained multi-frame emission evidence exceeding transition cost; single touches or `<1s` duel flickers (`min_spell_frames = 25`) are absorbed into surrounding control.

---

## 5. Benchmark Checks & Limitations

### Internal Verification (Strict Zero-Leakage)

| Stage / Pipeline | Public 5-min `spellF1` | Public 5-min `macroF1` | Alive/Dead `F1` | Team Acc | Full Match `spells/5min` | `<1s` Spells | `H1 / H2` Spells/5m |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_raw` | `0.0378` | `0.5972` | `0.8920` | `0.7784` | `92.02` | `51.8%` | `91.92 / 92.11` |
| `baseline_smooth125` | `0.4242` | `0.5812` | `0.8991` | `0.8854` | `18.79` | `18.1%` | `20.01 / 17.57` |
| `ablation4_step1_physics` | `0.5647` | `0.8785` | `0.9769` | `0.8913` | `15.11` | `0.0%` | `15.38 / 14.83` |
| `ablation5_learned_viterbi` | `0.6000` | `0.8805` | `0.9769` | `0.8941` | `15.11` | `0.0%` | `14.35 / 15.87` |
| **`ablation8_step3_em_hsmm` (Final)** | **`0.6000`** | **`0.9026`** | **`0.9865`** | **`0.8972`** | **`15.77`** | **`0.0%`** | **`15.48 / 16.06`** |

Submitted `sendiri aj.csv`: `199,319` frames (`H: 65,835 [33.0%]`, `A: 53,302 [26.7%]`, `D: 39,720 [19.9%]`, `X: 40,462 [20.3%]`), reporting `334` possessions with mean duration `14.3s`.

### Known Limitations & Future Fixes
1. **Short Interceptions & Rapid Turnovers (<2s):** Strong transition penalties (`pen_ha = -26`) eliminate false occlusion splits (`0.0% <1s` slivers) but occasionally merge genuine sub-2-second turnovers inside contested duels (`GT #11`/`#13`). Modeling explicit kick-to-reception velocity reversals with ball-tracking Kalman filtering would sharpen short-turnover boundaries.
2. **Boundary Lag Under Early/Late Smart Tagging Drift:** When automated defensive/offensive Smart Tagging events carry trailing polarity after an opponent regains possession in slow buildup, learned transition timing can shift boundaries by `1–4s` even when the winning team is eventually identified. Conditioning transition emissions on pass-receipt velocity divergence (`kick_impulse` aligned with `nw_rel_v`) rather than stationary phase tags is the highest-leverage next improvement.
3. **Sub-4-Second Stoppages:** Quick free kicks/throw-ins under ~4s without a sharp ball-absent period or collective walking drop remain hard to separate from slow tactical pauses using kinematics alone.

---

## 6. Exact Reproduction

Tested on **Python 3.12.6** with pinned dependencies in `pyproject.toml`, `uv.lock`, and `requirements.txt`:

```bash
# 1. Install pinned environment
uv sync --frozen
# (or: pip install -r requirements.txt)

# 2. Place raw tactical bundle under data/raw/tactical/ and convert to Parquet
uv run python scripts/convert_to_parquet.py --source tactical

# 3. Deterministically regenerate exact submission CSV
uv run python scripts/generate_submission.py --out "submissions/sendiri aj.csv" --model step3_hsmm

# 4. Validate submission format
uv run python reference/evaluation/validate_submission.py "submissions/sendiri aj.csv"
```
