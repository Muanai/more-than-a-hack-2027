# Exploratory Data Analysis (EDA) — Research Questions & Findings

The EDA notebook has one job: **turn the assets into a decision map for the submission**, not into a
tour of the files. Every question below must either change what we build or confirm a number we would
otherwise guess wrong.

Two things shape every question in this document:

1. **The ranking metric is `spellF1`** — possessions matched one-to-one by temporal IoU, averaged over
   thresholds 0.3 / 0.4 / 0.5 / 0.6 / 0.7. `macroF1` is the per-frame view and the first tie-break.
   The two disagree on purpose; `spellF1` is the one that decides the ranking.
2. **The public ground truth is 5 minutes, 17 possessions, 21 segments.** That is small. It is enough
   to check correctness and to understand the metric. It is not enough to choose a parameter, and the
   rules forbid using it that way. `EVALUATION.md` demonstrates the cost: the smoothing window that
   wins on the public window (`smooth125`, 0.4242) loses on the full match (`smooth50`, 0.2433).

Numbers already measured while scoping this document are marked **measured** and carry the date. They
are provisional: the notebook must re-derive them from the parquet assets and the official metric
module, and any disagreement is a finding in itself.

---

## 0. Ground rules

### 0.1 Evidence base — tag every answer

Every question is tagged with what it can honestly be answered from:

| tag | meaning | statistical weight |
|---|---|---|
| `[FULL]` | whole match, 199,319 frames. Needs no labels. | strong |
| `[GT-5MIN]` | only inside frames 11,185–18,684, where labels exist | 17 possessions / 21 segments / 4 dead-ball runs — **directional only** |
| `[DERIVED]` | arithmetic or controlled perturbation of the ground truth or of a submission, no model fitting | exact, but describes the metric, not football |

`[GT-5MIN]` answers must never be used to select a hyperparameter. If a `[GT-5MIN]` number suggests a
parameter, write down the candidate range and move the selection to a full-match criterion (possession
count plausibility, stability under perturbation, or agreement across halves).

### 0.2 Q0.1 — Is the public window a fair sample of the match? `[FULL]`

- **Metric / Measurement**: compare, for the 5-minute window vs the real graded blocks
  (11,185–90,651 and 110,808–190,197): ball-availability rate, mean tracked objects per frame, ball
  out-of-bounds rate, and the fraction of frames with fewer than 10 tracked players.
- **Why it matters / what it gates**: every `[GT-5MIN]` question inherits the window's biases. If the
  window is unrepresentative on a dimension, that dimension's `[GT-5MIN]` answer is directionally
  indicative at best, and the notebook must say so in the cell that reports it.
- **Decision rule**: if ball availability or tracked-object density differs by more than 10 relative
  percent between window and match, no `[GT-5MIN]` measurement on that dimension may inform a design
  decision without a full-match cross-check.
- **Measured (2026-10-31)**: ball availability **88.2%** in the window vs **59.3%** in the graded blocks
  (1st half 59.4%, 2nd half 59.3%). **The window is not representative on the single most important
  quality dimension.** Treat every window ball statistic as optimistic.
- **Answer / Findings**: *(To be answered — add tracked-player density and out-of-bounds rate)*

### 0.3 Q0.2 — What does `spellF1` actually reward? `[DERIVED]`

- **Metric / Measurement**: a controlled-perturbation sweep. Starting from the true labels, apply one
  distortion at a time and score with `evaluation/possession_metrics.py`:
  1. uniform boundary shift δ ∈ {0, 1, 3, 8, 25, 50} frames applied to every transition;
  2. split every spell into k equal pieces (k = 2, 3, 5, 10) — simulates jitter/no smoothing;
  3. merge every pair of adjacent same-team spells across an intervening dead-ball run — simulates a
     failed dead-ball call;
  4. drop all spells shorter than L ∈ {0, 0.5, 1, 2, 4} s — simulates minimum-duration filtering;
  5. flip the team label on a fraction f ∈ {0, 0.05, 0.1, 0.2} of spells.
  Plot `spellF1` against each distortion, and print the `macroF1` of the same distortion beside it.
- **Why it matters / what it gates**: this is the objective function's shape. It says which error is
  cheap and which is fatal, before we spend a day on feature engineering. The published examples cover
  only case 1 (a 1-frame offset is free, 1 s costs 0.35).
- **Decision rule**: the distortion with the steepest initial slope sets the priority order of the
  whole project. Expected — but must be measured — that case 2 (fragmentation) dominates, in which
  case no per-frame modelling work is worth doing before segmentation is under control.
- **Answer / Findings**: *(To be answered)*

### 0.4 Q0.3 — Which sub-problem binds: alive/dead, or home/away? `[GT-5MIN]`

- **Metric / Measurement**: for the four published baselines (`evaluation/results/baseline_*.csv`),
  report `alive_dead_f1`, `team_acc_both_alive`, `n_gt_spells` vs `n_pred_spells`, `fragmentation`,
  and `spell_f1_mean` side by side.
- **Why it matters / what it gates**: the baseline sweep moves `macroF1` by 0.04 and `spellF1` by a
  factor of five on the *same* algorithm. Decomposing that swing tells us whether smoothing is buying
  dead-ball stability or possession identity — they call for different fixes.
- **Decision rule**: whichever of `alive_dead_f1` / `team_acc_both_alive` moves least across the window
  sweep is not the bottleneck; do not spend feature work on it.
- **Answer / Findings**: *(To be answered)*

---

## 1. Regression assertions (not questions)

The notebook must assert these on load and stop if they fail. They are the cheapest defence against
the failure mode `validate_submission.py` exists to catch: a well-formed file built on a misaligned
index.

| check | expected | source |
|---|---|---|
| rows in tracking / homography / submission | 199,319 | DATA.md |
| frame index | tracking `frame_id` 1…199,319 → video frame `frame_id − 1` | DATA.md |
| homography `frame` | 0…199,318, already a submission row | DATA.md |
| graded blocks | 11,185–90,651 and 110,808–190,197 (158,857 frames) | DATA.md |
| public GT graded window | 11,185–18,684 (7,500 frames) | DATA.md / meta.json |
| GT label counts in window | H 3,454 · A 2,577 · D 1,469 | DATA.md |
| GT segments / possessions in window | 21 / 17 | meta.json |
| mean possession duration | 14.19 s | meta.json |
| ball available, whole match | 56.9% (113,488 frames) | DATA.md |
| ball available, graded blocks | 59.3% (94,214 frames) | DATA.md |
| track slots | 46 = 24 home + 22 away | ATD XML |
| mean tracked objects, graded frames | 20.5 | DATA.md |
| smart tagging instances | 638 | DATA.md |

---

## 2. Ground-truth anatomy `[GT-5MIN]`

### 2.1 Q2.1 — Label distribution during active play

- **Metric / Measurement**: counts and shares of `H`, `A`, `D` over the 7,500 graded window frames.
- **Why it matters / what it gates**: sets the class priors behind `macroF1` (unweighted over three
  classes) and gives the floor a constant guess scores.
- **Measured (2026-10-31)**: H 3,454 (**46.1%**), A 2,577 (**34.4%**), D 1,469 (**19.6%**). Dead ball is
  a fifth of graded frames but only 4 of 21 segments — the frame share and the segment share are
  different numbers, and `spellF1` scores the second.
- **Answer / Findings**: *(Assert against the table in §1, then report the three-class priors used
  downstream)*

### 2.2 Q2.2 — Duration distribution of real possessions

- **Metric / Measurement**: min / quartiles / median / mean / max of contiguous `H` and `A` runs, in
  frames and seconds; same for `D` runs. Report the full 17-value list, not just the summary.
- **Why it matters / what it gates**: possession length is the resolution at which a boundary error is
  affordable. It defines how precise our transitions must be, and it identifies the spells that are
  effectively unmatchable (see Q2.3).
- **Decision rule**: spells below ~1 s are treated as unmatchable in `spellF1` unless boundaries are
  frame-accurate; do not spend modelling effort trying to recover them, but do not merge them into
  neighbours either, since merging costs the neighbour's IoU.
- **Measured (2026-10-31)**: mean **14.19 s**, median **7.56 s**, min **0.76 s**, max **42.44 s**.
  Distribution is strongly right-skewed: 6 of 17 spells are under 6 s, 2 are under 2 s.
  Dead-ball runs: mean **14.69 s**, median **16.40 s**, min **5.52 s**, max **20.44 s** — dead-ball
  stretches are *longer and more uniform* than possessions.
- **Answer / Findings**: *(To be answered — paste the 17 durations into the notebook table; re-check
  against meta.json's 14.191 s mean)*

### 2.3 Q2.3 — What boundary accuracy does each spell length require? `[DERIVED]`

- **Metric / Measurement**: for a spell of length `T` and a symmetric boundary error `b` on both ends,
  `IoU = (T − 2b) / (T + 2b)`, so the maximum tolerable error at threshold `t` is
  **`b ≤ T(1 − t) / 2(1 + t)`**. Tabulate `b` for every measured spell length at t = 0.3 / 0.5 / 0.7,
  and count how many spells survive a boundary error of 0.5 s, 1 s and 2 s at each threshold.
- **Why it matters / what it gates**: turns the duration distribution into an engineering tolerance.
  This is what "how precise must the transition detector be" means numerically, and it is the input to
  choosing a smoothing / hysteresis design.
- **Measured (2026-10-31)**, from the 17 public spells:

  | spell (s) | b @ IoU 0.3 | b @ IoU 0.5 | b @ IoU 0.7 |
  |---|---|---|---|
  | 0.76 | 0.20 s | **0.13 s** | 0.07 s |
  | 1.32 | 0.36 s | 0.22 s | 0.12 s |
  | 2.84 | 0.76 s | 0.47 s | 0.25 s |
  | 3.24 | 0.87 s | 0.54 s | 0.29 s |
  | 3.48 | 0.94 s | 0.58 s | 0.31 s |
  | 3.60 | 0.97 s | 0.60 s | 0.32 s |
  | 6.64 | 1.79 s | 1.11 s | 0.59 s |
  | 7.44 | 2.00 s | 1.24 s | 0.66 s |
  | 7.56 | 2.04 s | 1.26 s | 0.67 s |
  | 8.08 | 2.18 s | 1.35 s | 0.71 s |
  | 11.84 | 3.19 s | 1.97 s | 1.04 s |
  | 16.04 | 4.32 s | 2.67 s | 1.42 s |
  | 20.00 | 5.38 s | 3.33 s | 1.76 s |
  | 28.60 | 7.70 s | 4.77 s | 2.52 s |
  | 35.72 | 9.62 s | 5.95 s | 3.15 s |
  | 41.64 | 11.21 s | 6.94 s | 3.67 s |
  | 42.44 | 11.43 s | 7.07 s | 3.74 s |

  Reading: at IoU 0.5 a 1 s boundary error costs the six shortest spells outright, and the 0.76 s spell
  needs **3-frame** accuracy. Two-thirds of the value sits in the long spells, where the tolerance is
  generous.
- **Answer / Findings**: *(To be answered — the notebook should compute this table, not hard-code it)*

### 2.4 Q2.4 — What do possession transitions look like?

- **Metric / Measurement**: transition-count matrix over the six ordered pairs `H→A`, `A→H`, `H→D`,
  `D→H`, `A→D`, `D→A`, plus the mean/min/max gap between consecutive transitions.
- **Why it matters / what it gates**: whether turnovers are usually direct (team to team) or mediated
  by a dead ball decides whether an alive/dead stage in the pipeline is a prerequisite for a team
  switch, or an independent output.
- **Measured (2026-10-31)**: 20 transitions in the window — `A→H` 6, `H→A` 6, `A→D` 3, `D→A` 2,
  `D→H` 2, `H→D` 1. **12 of 20 turnovers never touch `D`.** The hypothesis "dead ball mediates most
  turnovers" is false in the window. Also note the minimum gap between transitions is what a smoothing
  window can silently destroy — report it.
- **Answer / Findings**: *(To be answered — add the inter-transition gap distribution)*

### 2.5 Q2.5 — What do dead-ball runs look like?

- **Metric / Measurement**: number, duration, and boundary sharpness of `D` runs; for each, the state
  immediately before and after; and how often the run is symmetric around a set-piece restart.
- **Why it matters / what it gates**: `D` is a class in `macroF1` and a *barrier* for `spellF1` (a
  failed `D` merges two possessions and destroys both). The count of `D` runs bounds how much of the
  score is decided by dead-ball detection.
- **Measured (2026-10-31)**: 4 `D` runs at 12,851–12,989 (5.5 s), 15,889–16,400 (20.4 s),
  17,380–17,838 (18.3 s), 18,027–18,389 (14.5 s). Only two are bracketed by a change of team; two sit
  between an `A` spell and an `A` spell.
- **Answer / Findings**: *(To be answered)*

---

## 3. The ball signal

### 3.1 Q3.1 — How is the ball missing, and how does it come back? `[FULL]`

- **Metric / Measurement**: distribution of contiguous ball-`NaN` run lengths over the full match and
  over the graded blocks separately: count, median, p90, p99, max; the share of missing frames that
  fall inside gaps of ≤5 / ≤12 / ≤25 / ≤50 / ≤125 frames; and the gap-start/gap-end distribution by
  match phase.
- **Why it matters / what it gates**: decides whether gap-filling is a *micro-repair* problem or a
  *state-tracking* problem. These lead to completely different designs (interpolation vs a ball-state
  model / last-known-position prior), and only one of them is worth building.
- **Decision rule**: if fewer than ~10% of missing frames sit inside gaps of ≤12 frames, do not build
  an interpolation module at all.
- **Measured (2026-10-31)**: **186 gaps**, 85,831 missing frames. Median gap **230 frames (9.2 s)**,
  p90 **840**, p99 **5,535**, max **8,689 (5.8 min)**. Gaps ≤5 frames: 33 gaps covering **0.1%** of
  all missing frames; ≤25 frames: 43 gaps covering **0.2%**; ≤125 frames: 44 gaps covering **0.2%**.
  **Interpolation is worth at most 0.2% of the missing ball. The missingness is blocky, not flickery.**
  In the 5-minute window: only 6 gaps, median 140 frames.
- **Answer / Findings**: *(To be answered — add the gap-start phase breakdown, and check whether long
  gaps coincide with camera/halftime structure rather than with play)*

### 3.2 Q3.2 — What match state do we observe when the ball is missing? `[GT-5MIN]`

- **Metric / Measurement**: ball-availability rate conditioned on the GT label (`H`, `A`, `D`), inside
  the window; plus the reverse — the label distribution conditioned on ball present / absent.
- **Why it matters / what it gates**: tests the baseline's implicit assumption that missing ball ≈ dead
  ball. If missing ball happens during live play, treating absence as `D` destroys `H`/`A` recall and
  splits spells.
- **Measured (2026-10-31)**: ball present on **99.9%** of `H` frames, **85.0%** of `A` frames,
  **66.4%** of `D` frames. So the assumption is false in both directions: a third of dead-ball frames
  have a visible ball, and ball absence is *least* likely during home possession. The asymmetry between
  `H` and `A` (99.9% vs 85.0%) is almost certainly a camera/tracking artefact, not football — confirm
  or refute before using ball presence as a feature.
- **Answer / Findings**: *(To be answered — include the reverse conditional and the H/A asymmetry
  check against camera position)*

### 3.3 Q3.3 — Is the ball we do see physically plausible? `[FULL]`

- **Metric / Measurement**: derived per-frame ball displacement in metres (the ATD carries no usable
  speed — see Q3.4), its p50/p90/p99/p99.9 and maximum in m/s; the count of consecutive present-frame
  steps above 40 m/s; and the spatial distribution of those steps (near the pitch border? in the box?).
- **Why it matters / what it gates**: "wrong but present" ball is a different failure from "missing"
  ball, and it is the one that poisons proximity features. If a meaningful fraction of ball positions
  are physically impossible, a proximity model needs a plausibility gate before it sees the ball at all.
- **Decision rule**: if more than ~2% of consecutive steps exceed 40 m/s, add an explicit ball-speed
  gate as a feature, not as a post-hoc filter.
- **Measured (2026-10-31)**: p50 **5.5 m/s**, p90 **17.2**, p99 **50.3**, p99.9 **84.7**. No teleports
  (0 steps above 625 m/s), but ~1% of steps exceed 50 m/s, above any real ball speed. Median ball speed
  over the whole match is 5.5 m/s, which is low for a live ball — consistent with the tracker holding a
  stationary position while the ball is out of view.
- **Answer / Findings**: *(To be answered — this is a high-value question; localise the implausible
  steps)*

### 3.4 Q3.4 — Is the exported `ball_speed` column usable? `[FULL]`

- **Metric / Measurement**: non-`NaN` count of `ball_speed`, its frame range, its agreement with
  speed derived by differencing, and whether it is ever present on a frame where `ball_x` is `NaN` or
  vice versa.
- **Why it matters / what it gates**: a feature column that is present in the parquet but `NaN` on the
  frames you validate on produces a silent train/validate mismatch. This is a data-integration bug
  waiting to happen, not an analysis question.
- **Measured (2026-10-31)**: `ball_speed` non-`NaN` on **9,909 / 199,319 frames (5.0%)**, only within
  frames 54,835–187,821, and **0 frames inside the public window**. Unusable as a feature; derive speed
  from positions and document the decision.
- **Answer / Findings**: *(Assert in the notebook; add to the technical document as a dropped asset)*

---

## 4. Player-ball geometry and team attribution

### 4.1 Q4.1 — How do we know which team a track slot belongs to, and are slots reliable? `[FULL]`

- **Metric / Measurement**: slots per team from the ATD XML; per-slot active-frame count; number of
  simultaneous slots per team per frame; duplicate detection (two slots of the same team within, say,
  1 m of each other for more than N frames); short-lived slots (< 1 s of total presence); and how often
  the nearest player overall belongs to a duplicated slot.
- **Why it matters / what it gates**: every proximity feature depends on slot→team being correct and on
  the argmin not landing on a ghost track. It is a stated prerequisite for Q4.3 and the whole baseline
  approach, and nothing in the current asset documentation settles it.
- **Decision rule**: any slot with more than ~5% duplicate-overlap frames is excluded from
  nearest-player distances; document the exclusion rule rather than tuning it.
- **Measured (2026-10-31)**: **46 slots = 24 home + 22 away** for 22 players on the pitch — the home
  side carries two extra slots. Mean tracked objects on graded frames is 20.5, and the tracker
  occasionally exceeds 22 by duplicating someone.
- **Answer / Findings**: *(To be answered)*

### 4.2 Q4.2 — What does distance-to-ball look like during possession? `[GT-5MIN]`

- **Metric / Measurement**: distribution of nearest-home distance `d_H`, nearest-away distance `d_A`,
  and the margin `d_A − d_H`, conditioned on the GT label (`H` vs `A`), restricted to ball-present
  frames. Report medians and quartiles per class, in metres, using the 105.3 × 68.0 m pitch.
- **Why it matters / what it gates**: quantifies the physical boundary of "possession" and tells us
  whether a fixed distance threshold (1 m? 2 m? 4 m?) carries any signal at all, or whether only the
  *relative* margin does.
- **Decision rule**: if the two class-conditional distance distributions overlap by more than ~50% of
  their area, absolute distance is not a usable threshold and every feature must be relative
  (margin, rank, team-aggregate).
- **Measured (2026-10-31)**, unconditioned, window, ball present: nearest home median **3.27 m**
  (p90 9.15), nearest away median **5.49 m** (p90 16.87). The class-conditional split is still open.
- **Answer / Findings**: *(To be answered)*

### 4.3 Q4.3 — How often does the nearest player not belong to the possessing team? `[GT-5MIN]`

- **Metric / Measurement**: frame-level accuracy of `argmin(d_H, d_A)` against the GT label over
  ball-present alive frames; the same split by class; and accuracy bucketed by margin
  (`|d_A − d_H|` in `(0, 0.5]`, `(0.5, 2]`, `> 2` m). Then, separately, the number of times the
  nearest-team label *flips* between consecutive frames, compared with the true segment count.
- **Why it matters / what it gates**: gives the per-frame ceiling of pure proximity — the number the
  baseline is fighting — and, more importantly, separates two different failures: getting the team
  wrong (a per-frame problem) versus flickering (a temporal problem). They need different fixes, and
  the ranked metric only charges the second one heavily.
- **Measured (2026-10-31)**: accuracy **77.8%** overall (H 79.7%, A 74.9%) over 5,640 alive
  ball-present frames. By margin: `|margin| > 2 m` → **90.7%**; `0.5–2 m` → 65%; `< 0.5 m` → **56.9%**
  (n = 536, i.e. near coin-flip but only 9.5% of frames). Flicker: **136 nearest-team changes in 7,500
  frames against 21 true segments — 6.5× over-segmentation.** `baseline_raw` predicts 131 possessions
  (spellF1 0.038) while holding `macroF1` 0.597.
- **Answer / Findings**: *(To be answered)*

---

## 5. Temporal fragmentation — highest priority

Over-segmentation is the binding constraint on the ranked metric (§4.3), and it is not currently a
question in this document's model of the problem. These four questions exist to fix that.

### 5.1 Q5.1 — What does the raw signal's flicker look like? `[GT-5MIN]`

- **Metric / Measurement**: run-length distribution of the raw nearest-team signal over the window
  (ignoring ball-absent frames); number of runs, median/p10/p90 run length; share of runs shorter than
  6 / 12 / 25 / 50 frames; and the same for a ball-carrying rule (last team with the ball) rather than
  nearest-player.
- **Why it matters / what it gates**: 6.5× too many segments cannot be fixed by a better classifier —
  only by a temporal decision rule. The run-length distribution says how aggressive that rule must be
  and whether a median filter, a hysteresis band, or an explicit state machine is the right tool.
- **Decision rule**: if more than half of raw runs are shorter than 0.5 s, a minimum-dwell rule is
  mandatory before any team classification is even evaluated.
- **Answer / Findings**: *(To be answered)*

### 5.2 Q5.2 — How does `spellF1` respond to predicted possession count? `[DERIVED]`

- **Metric / Measurement**: sweep a single scrolling-median / minimum-dwell parameter over the raw
  proximity signal, and plot `spellF1` (and `macroF1`) against `n_pred_spells / n_gt_spells` and
  against mean predicted spell duration. Mark where the count crosses the true 17.
- **Why it matters / what it gates**: this is the curve the whole competition lives on. `n_pred/n_gt`
  is measurable on the **full match without labels** — it is the only honest, non-tuning proxy we have
  for "is my segmentation the right scale?". That makes it the primary model-selection signal.
- **Decision rule**: adopt `n_pred_spells` per 5 minutes (or mean spell duration) as the full-match
  selection criterion. **Do not** select the parameter by maximising public-window `spellF1` — the
  competition's own documentation shows that picks the wrong setting.
- **Answer / Findings**: *(To be answered — expect a sharp peak; record its width, since the plateau
  defines the safe parameter range)*

### 5.3 Q5.3 — Where do transitions actually get missed? `[GT-5MIN]`

- **Metric / Measurement**: from `evaluate()`'s `transitions` block at τ = 12 and 25 frames: transition
  precision and recall, `n_missed`, `n_spurious`, `n_ambiguous`, and the mean/median signed latency.
- **Why it matters / what it gates**: a large gap between τ=12 and τ=25 scores is a systematic lag
  detector — a bias no frame-level metric reveals. Signed latency tells us whether to shift the
  decision point earlier or later, which is a free score improvement if it is consistent.
- **Decision rule**: if mean latency exceeds ~6 frames in magnitude and is consistent in sign, correct
  it explicitly and re-measure; if it is small and zero-mean, leave it alone.
- **Answer / Findings**: *(To be answered)*

### 5.4 Q5.4 — What does an abstention (`X`) actually cost? `[DERIVED]`

- **Metric / Measurement**: inject `X` runs of length 5 / 25 / 50 frames into a baseline submission at
  three positions — inside a possession, at a true transition, and inside a dead-ball run — and measure
  the Δ`spellF1` and Δ`macroF1` for each. Also confirm the scorer's forward-fill behaviour on an `X` run
  at the very start of a half.
- **Why it matters / what it gates**: `spellF1` forward-fills `X` within a half, so abstention is free
  inside a possession and harmful across a hand-over. That asymmetry is a documented, exploitable
  lever for the final submission, and the current question set never mentions `X` at all.
- **Decision rule**: if the measured cost inside a possession is zero, the pipeline should emit `X`
  whenever its confidence is below threshold *and* it believes it is mid-possession; write the
  abstention policy explicitly rather than leaving it as a constant label.
- **Answer / Findings**: *(To be answered)*

---

## 6. Dead-ball signatures

### 6.1 Q6.1 — Does the ball leave the pitch during dead balls, and how useful is that? `[GT-5MIN]`

- **Metric / Measurement**: for ball-present frames, the rate of out-of-bounds coordinates
  (`x < 0 | x > 1 | y < 0 | y > 1`, and with a small margin) conditioned on each GT label. Report as
  **precision and recall** of "out-of-bounds ⇒ `D`", not as a correlation, plus the lead/lag between
  the ball crossing the line and the `D` run starting.
- **Why it matters / what it gates**: an out-of-bounds ball is an unambiguous dead-ball indicator, but
  its recall may be tiny. Precision/recall at a 5-second horizon is what tells us whether it belongs in
  a feature set or in a rules layer.
- **Measured (2026-10-31)**: among ball-present window frames — `H` **0.0%**, `A` **0.1%**,
  `D` **10.7%** out of bounds. High precision, roughly 11% recall: it fires almost only on dead ball,
  but misses 9 dead-ball frames in 10. Only 4.4% of all present ball frames are out of bounds.
- **Answer / Findings**: *(To be answered — add lead/lag and the margin sensitivity)*

### 6.2 Q6.2 — Do players slow down during dead balls? `[GT-5MIN]`

- **Metric / Measurement**: mean and p90 of player speeds (`p*_speed`, in m/s) per frame — averaged
  over active players and split by team — plus ball speed, conditioned on `H` / `A` / `D`. Also the
  frame-level separability of the best single threshold.
- **Why it matters / what it gates**: fouls, injuries and set-piece set-up produce a collective
  velocity drop. If it separates cleanly it is a cheap, ball-free dead-ball feature — valuable
  precisely on the frames where the ball is missing.
- **Decision rule**: a feature is worth building only if its best single-threshold separation is
  clearly above the `D` base rate (19.6% of frames).
- **Answer / Findings**: *(To be answered)*

---

## 7. Smart Tagging priors

### 7.1 Q7.1 — How dense is Smart Tagging coverage? `[FULL]`

- **Metric / Measurement**: share of video frames and of graded frames covered by at least one interval;
  per-code frame counts (not instance counts); overlap structure (how many codes are active at once,
  distribution); and coverage by match phase.
- **Why it matters / what it gates**: decides whether tagging is a dense per-frame feature or an
  occasional prior. The instance counts published in DATA.md do not answer this, because a `BUILD UP`
  and a `DEFENDING IN MIDDLE THIRD` are different lengths.
- **Measured (2026-10-31)**: coverage **139,587 frames = 70.0% of the video, 82.3% of graded frames**.
  Dense enough to be a per-frame feature. The heaviest codes are `BALL IN FINAL THIRD` (93 instances)
  and `PLAYERS IN THE BOX` (97), but by frame count the phases (`PROGRESSION`, `DEFENDING IN MIDDLE
  THIRD`) dominate — the two rankings disagree, which is itself the finding.
- **Answer / Findings**: *(To be answered — add the concurrency histogram)*

### 7.2 Q7.2 — Do set-piece boundaries align with dead-ball boundaries? `[GT-5MIN]`

- **Metric / Measurement**: **boundary agreement, not precision/recall.** For each `SET PIECES`
  instance, does its start fall within ±25 / ±50 frames of a `D → alive` transition, and does its end
  fall within ±25 / ±50 frames of an `alive → D` transition? Report the match rate per boundary and the
  signed offset distribution. Report raw overlap counts as context only.
- **Why it matters / what it gates**: set-piece intervals are fixed-length boxes around restarts, not
  dead-ball spans, so overlap-based precision/recall is undefined at this sample size. Boundary
  agreement is the question that can actually be answered, and it is the only version that informs a
  restart detector.
- **Measured (2026-10-31)**: `SET PIECES` has **59 instances in the match but 3 overlapping the
  window**. The three are: Kick Off 11,097–11,222 (`Team = N/A`), Free Kick 17,533–18,408
  (`Atletic Club`), Goal Kick 18,320–19,195 (`Barcelona`). Against the window's 4 `D` runs, the overlap
  counts are **0, 0, 1, 2** — the free kick spans *two* separate `D` runs with an `A` possession
  between them. Intervals are ~5 s boxes far wider than the dead-ball stretches they surround.
- **Decision rule**: `SET PIECES` may only be used as a restart *marker* (a candidate `D → alive`
  boundary), never copied across as a label span. If boundary agreement at ±50 frames is below ~50%,
  drop it from the feature set.
- **Answer / Findings**: *(To be answered)*

### 7.3 Q7.3 — Which codes are on-ball and which are off-ball? `[GT-5MIN]`

- **Metric / Measurement**: for each code, agreement between its `Team` tag and the GT label over the
  tagged frames. Then — and this is the question — test the **implied possessor** hypothesis: for
  offensive codes (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`,
  `LONG BALL`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`) the expected possessor is the tagged team;
  for defensive codes (`DEFENDING IN * THIRD`, `DEFENSIVE TRANSITION`) it is the *other* team. Report
  agreement under both readings and the difference between them.
- **Why it matters / what it gates**: the current version of this question pools codes whose `Team` tag
  means opposite things, so its average is guaranteed to be meaningless. Splitting by on-ball/off-ball
  converts the tagging into a usable, sign-corrected team prior — which is exactly the free evidence
  DATA.md says it should be.
- **Measured (2026-10-31)**, tag team == GT label, window:

  | code | frames | tag team == GT |
  |---|---|---|
  | `BUILD UP` | 295 | **1.000** |
  | `PROGRESSION` | 2,707 | 0.726 |
  | `ATTACKING TRANSITION` | 277 | 0.671 |
  | `BALL IN FINAL THIRD` | 3,039 | 0.537 |
  | `DEFENSIVE TRANSITION` | 277 | **0.329** |
  | `DEFENDING IN MIDDLE THIRD` | 2,707 | **0.274** |

  The defensive codes agree with the GT *label* only because some frames inside a defending interval
  belong to the defending team's own possession; the complement is what carries the signal. The pooled
  average over all codes would land near 0.55 and describe nothing.
- **Decision rule**: use a per-code sign, derived here and frozen; any code whose implied-possessor
  agreement is below ~60% does not enter the feature set.
- **Answer / Findings**: *(To be answered — compute the complement agreement for the defensive codes,
  and the four remaining codes)*

### 7.4 Q7.4 — How much do we lose to `Team = N/A`? `[FULL]`

- **Metric / Measurement**: count of instances with `Team = N/A` by code, and the frame coverage those
  instances represent; specifically confirm whether **all 26 throw-ins** are `N/A` as documented.
- **Why it matters / what it gates**: DATA.md warns that every throw-in is untagged, "exactly the
  restarts where knowing the side would be worth the most". If that holds, the team prior has a
  structural hole exactly at `D → alive` boundaries — the highest-value frames in the whole problem.
- **Decision rule**: if throw-ins are all `N/A`, do not attempt to fill them from tags; resolve them
  from the tracking's last-touch geometry instead, and record the limitation in the technical document.
- **Answer / Findings**: *(To be answered)*

---

## 8. Coverage and geometry checks

### 8.1 Q8.1 — Is player tracking density stable? `[FULL]`

- **Metric / Measurement**: distribution of non-`NaN` tracked slots per frame, by match phase (pre-match,
  1st half, halftime, 2nd half, post-match); share of frames with < 10 or > 22 tracked objects.
- **Why it matters / what it gates**: collective features (team centroid, pressing line, compactness)
  are biased by missing players, and the bias is not uniform across the match.
- **Measured (2026-10-31)**: mean 20.5 over graded frames (DATA.md); confirm here.
- **Answer / Findings**: *(To be answered)*

### 8.2 Q8.2 — Does the coordinate frame flip at halftime? `[FULL]`

- **Metric / Measurement**: team-mean `x` position per half, and the sign of the change; test whether
  normalising by attack direction is required before any "own half / final third" feature.
- **Why it matters / what it gates**: teams switch ends. A feature like "defensive third" silently
  inverts for one team in one half if the frame is not attack-direction-normalised. One cell to check,
  expensive to find later.
- **Decision rule**: if the team-mean `x` ordering reverses across halves, every spatial feature must be
  expressed in attack-relative coordinates.
- **Answer / Findings**: *(To be answered)*

### 8.3 Q8.3 — (optional, low priority) Tactical homography quality `[FULL]`

- **Metric / Measurement**: median and p95 `reprojection_error` by phase.
- **Status**: **demoted.** The tactical homography is 100% valid and the ATD already arrives in
  normalised pitch coordinates; reprojection error only matters if we read video pixels. Median is 0.24
  during play and ~1.5 at halftime, changing nothing. Answer this only if the pipeline starts using the
  video, and otherwise record it in the technical document as a checked-and-dropped asset.

---

## 9. Considered and deliberately not asked

| question | why it is out |
|---|---|
| "How much missing ball can interpolation restore?" | Measured: at most **0.2%** of missing frames sit in gaps ≤ 25 frames. The premise (micro-flicker) is falsified; replaced by Q3.1. |
| "What is the precision/recall of `SET PIECES` against `D`?" | **3 instances** overlap the window; the intervals are fixed 5 s boxes that span multiple `D` runs. Undefined at this sample size; replaced by Q7.2 (boundary agreement). |
| "What is the correlation between out-of-bounds and `D`?" | A correlation on a rare binary event describes nothing; replaced by precision/recall in Q6.1. |
| "What smoothing window is best?" | Selecting it on the public window is both forbidden and empirically wrong (`smooth125` wins publicly, `smooth50` wins on the match). Replaced by the full-match `n_pred_spells` criterion in Q5.2. |
| "How does the broadcast feed help?" | Not frame-aligned to the tactical video, 65.8% valid homography, no mapping table. Out of scope until the tactical-only baseline is solid; revisit only if there is time left. |
| "What happens on the 40,462 ungraded frames?" | Nothing — they are never scored, and `X` there is free. One line in the submission writer, not an EDA question. |

---

## 10. Notebook build order

A marimo notebook that follows the document, cheapest decision first:

1. **Load + assert** — §1 table. Nothing runs until every assertion passes.
2. **Metric map** — Q0.2 perturbation sweep. Pure GT arithmetic; no model. Produces the priority order.
3. **GT anatomy** — Q2.1, Q2.2, Q2.4, Q2.5, and the Q2.3 tolerance table.
4. **Ball quality** — Q3.1, Q3.3, Q3.4, then Q3.2 conditioned on labels.
5. **Geometry** — Q4.1, Q4.2, Q4.3, including the flicker count.
6. **Fragmentation** — Q5.1, Q5.2, Q5.3, then Q5.4.
7. **Dead ball** — Q6.1, Q6.2.
8. **Smart Tagging** — Q7.1, Q7.3, Q7.4, then Q7.2.
9. **Coverage** — Q8.1, Q8.2. (Q8.3 only if video is used.)

Rules of construction:

- Read the parquet assets, not the raw XML/TXT, so a cell re-run costs seconds.
- One `[GT-5MIN]` cell that sets `evidence = "GT-5MIN"` must print that tag in its output, so nobody
  mistakes a directional number for a full-match one.
- Never write a conclusion into the document that a cell cannot regenerate.
