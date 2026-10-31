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
- **Answer / Findings (`[FULL]`)**:
  - **Ball availability**: **88.20%** in the 5-min window vs **59.31%** in graded play (1st Half `59.35%`, 2nd Half `59.27%`, Full Video `56.94%`) — a **+48.7% relative discrepancy**.
  - **Tracked-player density**: Mean **21.57** (median 22) in the window vs **20.49** (median 22) in graded play (`20.12` in 1st Half, `20.85` in 2nd Half).
  - **Low-density & duplicate frames**: Frames with `<10` tracked players occur on **0.00%** of the window vs **3.05%** of graded play (`4.18%` in 1st Half, `1.91%` in 2nd Half). Frames with `>22` players occur on **0.05%** of the window vs **2.36%** of graded play.
  - **Ball out-of-bounds rate (among present frames)**: **1.60%** in the window vs **4.19%** in graded play (`2.72%` in 1st Half, `5.67%` in 2nd Half) — a **2.6× discrepancy**.
  - **Gate triggered**: Both ball availability and OOB rate exceed the 10% relative difference threshold. The 5-min public window is an unusually clean opening stretch; no hyperparameter (especially smoothing width or missing-ball handling) may be tuned on it directly.

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
- **Answer / Findings (`[DERIVED]`)**:
  - **1. Boundary shift ($\delta$)**: $\delta = 1\text{f}$ and $\delta = 3\text{f}$ (`0.12s`) cost **0.0000** in `spellF1` (`1.0000`); $\delta = 8\text{f}$ (`0.32s`) → `0.9529`; $\delta = 25\text{f}$ (`1.00s`) → `0.7765` (`macroF1 = 0.9331`); $\delta = 50\text{f}$ (`2.00s`) → `0.5882`.
  - **2. Fragmentation ($k$ splits per spell)**: Splitting each spell into just **$k = 2$** pieces via a 1-frame `D` blip crashes `spellF1` from **1.0000 → 0.3137** (`spellF1@0.5 = 0.2353`) while `macroF1` stays at **0.9971**! At $k = 3$, `spellF1 = 0.1000` (`macroF1 = 0.9942`); at $k = 5$, **`spellF1 = 0.0000`** (`macroF1 = 0.9886`).
  - **3. Failed `D` call (merging across `D`)**: Merging same-team spells across `D` (1 `A–D–A` run in the window) keeps `spellF1 = 0.9455` (`macroF1 = 0.8997`). Even erasing all 4 `D` runs completely leaves `spellF1 = 0.8121` (`macroF1 = 0.5913`).
  - **4. Dropping short spells ($< L\text{ s} \to \text{D}$)**: $L = 0.5\text{s} \to 1.0000$; $L = 1.0\text{s} \to 0.9697$ (`16` spells); $L = 2.0\text{s} \to 0.9375$ (`15` spells); $L = 4.0\text{s} \to 0.7857$ (`11` spells).
  - **5. Flipping team on fraction $f$ of spells**: $f = 0.05$ (`1/17`) → `0.9455`; $f = 0.10$ (`2/17`) → `0.8485`; $f = 0.20$ (`3/17`) → `0.7500`.
  - **Priority Order**: **Fragmentation (Case 2) is fatal** (steepest slope by far: `-0.686` for a single split per spell). Temporal segmentation / anti-flicker filtering is strictly priority #1; team attribution per spell is #2; dead-ball boundary precision is #3; recovering sub-second spells ($<1\text{s}$) is lowest priority.

### 0.4 Q0.3 — Which sub-problem binds: alive/dead, or home/away? `[GT-5MIN]`

- **Metric / Measurement**: for the four published baselines (`evaluation/results/baseline_*.csv`),
  report `alive_dead_f1`, `team_acc_both_alive`, `n_gt_spells` vs `n_pred_spells`, `fragmentation`,
  and `spell_f1_mean` side by side.
- **Why it matters / what it gates**: the baseline sweep moves `macroF1` by 0.04 and `spellF1` by a
  factor of five on the *same* algorithm. Decomposing that swing tells us whether smoothing is buying
  dead-ball stability or possession identity — they call for different fixes.
- **Decision rule**: whichever of `alive_dead_f1` / `team_acc_both_alive` moves least across the window
  sweep is not the bottleneck; do not spend feature work on it.
- **Answer / Findings (`[GT-5MIN]`)**:
  | Baseline | `spellF1` | `spellF1@0.5` | `macroF1` | `alive_dead_f1` | `team_acc_both_alive` | `n_pred_spells` | `fragmentation` |
  |---|---|---|---|---|---|---|---|
  | `baseline_raw` | 0.0378 | 0.0270 | 0.5972 | 0.8920 | 0.7784 | 131 | 6.52 |
  | `baseline_smooth25` | 0.2868 | 0.3019 | 0.6200 | 0.8974 | 0.8271 | 36 | 1.90 |
  | `baseline_smooth50` | 0.3256 | 0.3256 | 0.6207 | 0.8990 | 0.8582 | 26 | 1.43 |
  | `baseline_smooth125` | 0.4242 | 0.4242 | 0.5812 | 0.8991 | 0.8854 | 16 | 0.81 |
  - **Finding**: `alive_dead_f1` is virtually constant across the entire sweep (`0.8920 → 0.8991`, total span `0.0071`). The 11.2× gain in `spellF1` comes from eliminating `H ↔ A` flicker (`n_pred_spells` dropping from `131` to `16`) and lifting `team_acc_both_alive` from `0.7784` to `0.8854`. (Note however from Q0.1 that on the full match, ball visibility is 59.3% rather than 88.2%, so naive missing-ball = `D` degrades much more outside the window).

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Frame vs Segment shares**: `H = 3,454 (46.05%)` frames vs `8/21 (38.10%)` segments; `A = 2,577 (34.36%)` frames vs `9/21 (42.86%)` segments; `D = 1,469 (19.59%)` frames vs `4/21 (19.05%)` segments.
  - **Constant-guess baselines**: Predicting all-`H` scores `macroF1 = 0.2102`, all-`A` scores `0.1705`, and all-`D` scores `0.1092`; all three score `0.0000` on `spellF1` (`@0.3..0.7` and `@0.5`).

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Summary by class**:
    - `H` (8 spells): min `0.76s` (`19f`), Q25 `2.94s`, median `9.96s` (`249f`), mean `17.27s` (`431.8f`), Q75 `31.86s`, max `42.44s` (`1,061f`).
    - `A` (9 spells): min `2.84s` (`71f`), Q25 `3.60s`, median `7.44s` (`186f`), mean `11.45s` (`286.3f`), Q75 `16.04s`, max `35.72s` (`893f`).
    - Combined alive (17 spells): min `0.76s`, Q25 `3.48s`, median `7.56s`, mean **`14.191s`** (matches `meta.json`), Q75 `20.00s`, max `42.44s`.
    - `D` (4 runs): min `5.52s` (`138f`), Q25 `12.24s`, median `16.40s` (`410f`), mean `14.69s` (`367.2f`), Q75 `18.85s`, max `20.44s` (`511f`).
  - **Full 17-spell list (sorted by duration, with chronological Spell # and frame span)**:
    1. `0.76s` (`19f`, Spell #10, `H`, `15,346..15,365`)
    2. `1.32s` (`33f`, Spell #12, `H`, `15,766..15,799`)
    3. `2.84s` (`71f`, Spell #3, `A`, `11,553..11,624`)
    4. `3.24s` (`81f`, Spell #7, `A`, `13,704..13,785`)
    5. `3.48s` (`87f`, Spell #15, `H`, `17,293..17,380`)
    6. `3.60s` (`90f`, Spell #13, `A`, `15,799..15,889`)
    7. `6.64s` (`166f`, Spell #1, `A`, `11,185..11,351`)
    8. `7.44s` (`186f`, Spell #5, `A`, `12,665..12,851`)
    9. `7.56s` (`189f`, Spell #16, `A`, `17,838..18,027`)
    10. `8.08s` (`202f`, Spell #2, `H`, `11,351..11,553`)
    11. `11.84s` (`296f`, Spell #17, `H`, `18,389..18,685`)
    12. `16.04s` (`401f`, Spell #11, `A`, `15,365..15,766`)
    13. `20.00s` (`500f`, Spell #9, `A`, `14,846..15,346`)
    14. `28.60s` (`715f`, Spell #6, `H`, `12,989..13,704`)
    15. `35.72s` (`893f`, Spell #14, `A`, `16,400..17,293`)
    16. `41.64s` (`1,041f`, Spell #4, `H`, `11,624..12,665`)
    17. `42.44s` (`1,061f`, Spell #8, `H`, `13,785..14,846`)

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
- **Answer / Findings (`[DERIVED]`)**:
  - **Spell survival counts under symmetric boundary error $b \in \{0.5\text{s}, 1.0\text{s}, 2.0\text{s}\}$**:
    - `@ IoU 0.3`: `15/17 (88.2%)` survive `0.5s`, `11/17 (64.7%)` survive `1.0s`, `10/17 (58.8%)` survive `2.0s`.
    - `@ IoU 0.5`: `14/17 (82.4%)` survive `0.5s`, `11/17 (64.7%)` survive `1.0s`, `6/17 (35.3%)` survive `2.0s`.
    - `@ IoU 0.7`: `11/17 (64.7%)` survive `0.5s`, `7/17 (41.2%)` survive `1.0s`, `4/17 (23.5%)` survive `2.0s`.

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Transition counts**: `A→H: 6 (30%)`, `H→A: 6 (30%)`, `A→D: 3 (15%)`, `D→A: 2 (10%)`, `D→H: 2 (10%)`, `H→D: 1 (5%)`. Direct live-play turnovers account for `60%` of all state transitions.
  - **Inter-transition gap distribution** (`19` intervals between the `20` transition points): min **`19f (0.76s)`**, Q25 **`88f (3.54s)`**, median **`202f (8.08s)`**, mean **`370.4f (14.82s)`**, Q75 **`479f (19.16s)`**, max **`1,061f (42.44s)`**.

### 2.5 Q2.5 — What do dead-ball runs look like?

- **Metric / Measurement**: number, duration, and boundary sharpness of `D` runs; for each, the state
  immediately before and after; and how often the run is symmetric around a set-piece restart.
- **Why it matters / what it gates**: `D` is a class in `macroF1` and a *barrier* for `spellF1` (a
  failed `D` merges two possessions and destroys both). The count of `D` runs bounds how much of the
  score is decided by dead-ball detection.
- **Measured (2026-10-31)**: 4 `D` runs at 12,851–12,989 (5.5 s), 15,889–16,400 (20.4 s),
  17,380–17,838 (18.3 s), 18,027–18,389 (14.5 s).
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Correction to initial visual inspection**: **3 of the 4 `D` runs** mediate a change of team, and **only 1** sits between same-team (`A → D → A`) spells:
    1. `12,851..12,989` (`138f / 5.52s`): `A → D → H` (**Turnover**; ball visible on `95.7%` of frames)
    2. `15,889..16,400` (`511f / 20.44s`): `A → D → A` (**Same-team retain**; ball visible on `24.1%` of frames)
    3. `17,380..17,838` (`458f / 18.32s`): `H → D → A` (**Turnover**; ball visible on `85.6%` of frames)
    4. `18,027..18,389` (`362f / 14.48s`): `A → D → H` (**Turnover**; ball visible on `89.0%` of frames)
  - Because `15,889..16,400` is the only same-team `D` barrier, failing to call `D` while keeping `H`/`A` labels right only merges that single `A–D–A` pair (confirming Q0.2's finding that `spellF1` stays at `0.9455`).

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
- **Answer / Findings (`[FULL]`)**:
  - **Gap-start phase breakdown**:
    - `1st Half (11,185–90,651)`: `68` gaps, `32,659` missing frames, median **`293f (11.72s)`**, p90 `1,073f (42.9s)`, max `3,466f (138.64s)`.
    - `2nd Half (110,808–190,197)`: `81` gaps, `32,634` missing frames, median **`260f (10.40s)`**, p90 `639f (25.6s)`, max `5,537f (221.48s)`.
    - `Halftime`: `31` gaps (`838` missing frames, median `4f`, max `523f`).
    - `Pre-match`: `5` gaps (`11,011` missing frames, max `5,529f`); `Post-match`: `1` gap (`8,689` missing frames / `347.56s`).
  - **Graded-play cumulative gap share** (`149` gaps, `65,293` missing frames starting in graded halves):
    - `≤ 5f (0.20s)`: `13` gaps (`29` frames, **`0.044%`** of graded missing frames)
    - `≤ 12f (0.48s)`: `15` gaps (`51` frames, **`0.078%`**)
    - `≤ 25f (1.00s)`: `16` gaps (`73` frames, **`0.112%`**)
    - `≤ 50f (2.00s)`: `16` gaps (`73` frames, **`0.112%`**)
    - `≤ 125f (5.00s)`: `38` gaps (`2,358` frames, **`3.611%`**)
  - **Decision outcome**: `0.078% ≪ 10%` threshold — **do not build an interpolation module**; treat ball absence as a persistent state regime requiring player-context and carry-forward priors.

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Forward conditional $P(\text{Ball} \mid \text{GT})$**: `H = 99.86%` (`3,449/3,454` present), `A = 85.02%` (`2,191/2,577`), `D = 66.37%` (`975/1,469`).
  - **Reverse conditional $P(\text{GT} \mid \text{Ball})$**:
    - Given **Ball Present** (`6,615f`): `H = 52.14%`, `A = 33.12%`, `D = 14.74%`.
    - Given **Ball Absent** (`885f`): `D = 55.82%` (`494f`), `A = 43.62%` (`386f`), `H = 0.56%` (`5f`). Nearly **44% of missing-ball frames in the window are live Away possession**, proving that `ball_nan ⇒ D` is structurally broken.
  - **Camera / spatial asymmetry check**: When the ball is visible in the window, mean pitch `y` is **`0.709` during `A` possession** (far touchline from the main camera) vs **`0.445` during `H` possession** (`mean(x)` is `0.620` for `H` vs `0.565` for `A`). The `H` vs `A` dropout gap is a far-side camera/tracker occlusion artefact; raw `ball_present` must never be used as a team-discriminative feature.

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
- **Answer / Findings (`[FULL]`)**:
  - **Consecutive ball-present step distribution** (`113,303` steps): `p50 = 5.54 m/s`, `p90 = 17.23 m/s`, `p99 = 50.33 m/s`, `p99.9 = 84.66 m/s`, `max = 133.99 m/s`.
  - **Implausible step counts & localisation**:
    - Steps `> 40 m/s`: **`2,043` (`1.80%` of steps)**; steps `> 50 m/s`: **`1,158` (`1.02%`)**.
    - Spatial localisation of `> 40 m/s` steps: only **`8.4%` (`171`)** occur within 5% of the pitch border and **`1.7%` (`35`)** inside penalty boxes — the remaining ~90% happen in open play when the tracker snaps from a stale held position to the re-acquired ball.
  - **Decision outcome**: At `1.80%` (right at the `~2%` boundary) plus frequent stationary-ball holding during occlusion, include derived ball speed and acceleration (and a `> 40 m/s` jump indicator) directly as features so downstream models down-weight proximity during tracker snaps/stalls.

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
- **Answer / Findings (`[FULL]`)**:
  - Asserted in `notebooks/eda.py`: `ball_speed` is non-`NaN` on `9,909 / 199,319` frames (`4.97%`), restricted to frames `54,835..187,821`, **`0` frames in the public window**, `0` frames where `ball_x` is `NaN`, and correlates at **`-0.0975`** with finite-difference ball speed where present. **Dropped permanently from all feature pipelines.**

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
- **Answer / Findings (`[FULL]`)**:
  - **Inactive / pre-post-only slots**: Exactly **3 of the 46 slots have `0` active frames during the `158,857` graded frames**:
    - `Track 22` (`p22`, Home): `44` active frames total (`0` graded), `6.82%` duplicate overlap (`< 1.0 m` same-team).
    - `Track 23` (`p23`, Home): `107` active frames total (`0` graded), **`66.36%`** duplicate overlap.
    - `Track 44` (`p44`, Away): `0` active frames across the entire video.
  - **Graded-active slots (`43` slots = `22` Home + `21` Away)**: Every single graded-active slot has `< 5%` duplicate overlap (the highest are `Track 13` at `4.53%` and `Track 5` at `4.40%`). On graded ball-present frames, the nearest player to the ball belongs to a currently duplicated (`< 1.0 m` same-team) slot on **`2.99%`** of frames.
  - **Decision outcome**: Exclude `p22`, `p23`, and `p44` by the documented rule (`0` graded frames / `> 5%` duplicate rate); retain all `43` graded-active slots (`22 H + 21 A`).

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Class-conditional distance distributions on `6,615` ball-present window frames**:

    | GT Class | Frames | `d_H` p25 / Med / p75 (m) | `d_A` p25 / Med / p75 (m) | `min(d_H, d_A)` Med / p90 (m) | Margin `d_A − d_H` p25 / Med / p75 (m) |
    |---|---:|---|---|---|---|
    | `H` | 3,449 | **0.91 / 1.90 / 4.60** | 3.97 / 6.82 / 9.56 | 1.64 / 5.86 | **+0.48 / +3.44 / +7.54** |
    | `A` | 2,191 | 2.87 / 4.75 / 7.90 | **0.78 / 2.11 / 4.98** | 1.68 / 6.35 | **−4.44 / −1.82 / +0.01** |
    | `D` | 975 | 1.36 / 4.75 / 8.04 | 8.89 / 16.86 / 21.43 | 2.89 / 9.68 | +5.11 / +11.38 / +17.50 |

  - **Decision outcome**: During `H` and `A`, the possessing team's median distance is ~`2.0 m` (`p75 ≈ 4.8 m`, when the ball is in flight between passes), heavily overlapping the defending team's `p25` (`2.87–3.97 m`). Absolute distance thresholds fail during passes; relative features (`d_A − d_H`, `(d_A − d_H)/(d_A + d_H)`, top-$k$ team distance ranks) are mandatory.

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Accuracy breakdown on `5,640` alive ball-present frames**:
    - Overall `argmin(d_H, d_A)` accuracy: **`77.84%`** (`H` only: **`79.73%`** on `3,449f`; `A` only: **`74.85%`** on `2,191f`).
    - By margin `|d_A − d_H|`:
      - `≤ 0.5 m` (`536f`, `9.50%` of alive ball-present): **`56.90%`** accuracy (contested duels / close marking — effectively coin-flip).
      - `0.5–2.0 m` (`1,337f`, `23.71%`): **`65.45%`** accuracy.
      - `> 2.0 m` (`3,767f`, `66.79%`): **`85.21%`** accuracy.
  - **Flicker count**: On consecutive ball-present frames, `argmin(d_H, d_A)` flips **`127` times** (`128` contiguous team runs), and **`136` total label transitions** when including `D` gaps — a **6.8× transition excess** over the `20` true transitions (`21` segments).

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Raw `argmin(d_H, d_A)` on ball-present frames (`128` runs)**:
    - Run lengths: `p10 = 5.0f (0.20s)`, **`median = 18.5f (0.74s)`**, `mean = 51.7f (2.07s)`, `p90 = 124.2f (4.97s)`.
    - Share of short runs: `< 6f (0.24s) = 12.5%`, `< 12f (0.48s) = 32.0%`, **`< 25f (1.00s) = 54.7%`**, **`< 50f (2.00s) = 68.0%`**.
  - **Last-team carry-forward across all `7,500` frames (`129` runs)**:
    - Run lengths: `p10 = 5.0f (0.20s)`, **`median = 19.0f (0.76s)`**, `mean = 58.1f (2.33s)`, `p90 = 149.4f (5.98s)`.
    - Share of short runs: `< 6f = 12.4%`, `< 12f = 32.6%`, **`< 25f = 54.3%`**, **`< 50f = 66.7%`**.
  - **Decision outcome**: Nearly a third of raw runs are `< 0.5s` and **over half (`54.7%`) are `< 1.0s`** (triggered when a pass travels past a defender or during a brief duel). A temporal state-machine / hysteresis / minimum-dwell decoder is mandatory.

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
- **Answer / Findings (`[DERIVED]`)**:
  - **Majority-vote half-window sweep $w \in \{0..200\}$ (`window = 2w+1` frames)**:

    | $w$ (frames) | Window `2w+1` (s) | Window `spellF1` | Window `macroF1` | Window `n_pred_spells` | Full-Match Spells / 5-min | Full-Match Mean Spell (s) |
    |---:|---:|---:|---:|---:|---:|---:|
    | 0 | 0.04s | 0.0378 | 0.5973 | 131 | 92.0 | 1.98s |
    | 6 | 0.52s | 0.1463 | 0.6134 | 65 | 55.6 | 3.31s |
    | 12 | 1.00s | 0.2424 | 0.6197 | 49 | 44.3 | 4.18s |
    | 25 | 2.04s | 0.2868 | 0.6258 | 36 | 35.2 | 5.41s |
    | 50 | 4.04s | 0.3256 | 0.6207 | 26 | 28.0 | 7.24s |
    | 75 | 6.04s | 0.4000 | 0.6034 | **18** | 24.6 | 8.74s |
    | 100 | 8.04s | **0.4485** | 0.5976 | **16** | 21.5 | 10.47s |
    | 125 | 10.04s | 0.4242 | 0.5812 | 16 | 18.8 | 12.40s |
    | 150 | 12.04s | 0.4375 | 0.5665 | 15 | **17.2** | **13.95s** |
    | 200 | 16.04s | 0.4000 | 0.5567 | 13 | 14.9 | 16.57s |

  - **Key architectural takeaway**: In the public window, `n_pred_spells` crosses `17` between `w = 75` (`18` spells) and `w = 100` (`16` spells). However, in the baseline's naive majority voter, `D` is only emitted when *every* frame in `2w+1` has a missing ball—so large `w` erases real dead-ball stoppages (`macroF1` peaks at `w = 25` and drops steadily for `w ≥ 50`). **Decoupling `Alive vs D` segmentation from `H vs A` spell smoothing** allows us to reach the target full-match scale (~`16–19` spells per 5 minutes, mean duration ~`12–15s`) without erasing dead balls.

### 5.3 Q5.3 — Where do transitions actually get missed? `[GT-5MIN]`

- **Metric / Measurement**: from `evaluate()`'s `transitions` block at τ = 12 and 25 frames: transition
  precision and recall, `n_missed`, `n_spurious`, `n_ambiguous`, and the mean/median signed latency.
- **Why it matters / what it gates**: a large gap between τ=12 and τ=25 scores is a systematic lag
  detector — a bias no frame-level metric reveals. Signed latency tells us whether to shift the
  decision point earlier or later, which is a free score improvement if it is consistent.
- **Decision rule**: if mean latency exceeds ~6 frames in magnitude and is consistent in sign, correct
  it explicitly and re-measure; if it is small and zero-mean, leave it alone.
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Transition evaluation across baselines (`20` true GT transitions)**:
    - `baseline_raw`: `@τ=12` `P=0.037, R=0.250, F1=0.064` (`5` matched, `15` missed, `131` spurious, mean latency `-1.60f`); `@τ=25` `P=0.081, R=0.550, F1=0.141` (`11` matched, `125` spurious, `2` ambiguous, mean latency `-3.09f`, median `-4.0f`).
    - `baseline_smooth25`: `@τ=12` `P=0.128, R=0.250, F1=0.169` (`5` matched, `34` spurious, mean latency `+5.00f`, median `+7.0f`); `@τ=25` `P=0.154, R=0.300, F1=0.203` (`6` matched, `33` spurious, mean latency `+8.17f`, median `+9.0f`).
    - `baseline_smooth50`: `@τ=12` & `@τ=25` both `P=0.069, R=0.100, F1=0.082` (`2` matched, `18` missed, `27` spurious, mean latency `+4.00f`).
    - `baseline_smooth125`: `@τ=12` `P=0.125, R=0.100, F1=0.111` (`2` matched, `14` spurious, mean latency `-0.50f`); `@τ=25` `P=0.188, R=0.150, F1=0.167` (`3` matched, `13` spurious, mean latency `-7.67f`).
  - **Finding**: Centered window smoothing does not introduce a uniform one-way phase lag; rather, `70–90%` of true transitions are missed within `±25f` because passes and dead-ball restarts take more than `1s` for nearest-player geometry to cross over.

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
- **Answer / Findings (`[DERIVED]`)**:
  - **Empirical `X`-injection sweep (`L ∈ {5, 25, 50}` frames)**:
    - **Mid-possession (`12,000`) & Inside `D` (`16,000`)**: `ΔspellF1 = 0.0000` for all `L` (because `_forward_fill_x()` replaces `X` with the preceding non-`X` label), while `ΔmacroF1` is **`-0.0002` (`5f`), `-0.0012` (`25f`), `-0.0023` (`50f`)** on GT and **`-0.0003` to `-0.0058`** on `smooth50` (because `macroF1` counts `X` as a false negative on the true class).
    - **Across a true transition (`12,665`)**: Shifting the transition by `L` frames via `X` costs `ΔmacroF1 = -0.0002` to `-0.0022` (`ΔspellF1 = 0.0000` on a `41.6s → 7.4s` boundary, though fatal on short spells).
    - **Start of half (`11,185`)**: Leading `X` frames at the start of a graded half cannot be forward-filled and remain `'X'` in `_extract_spells`, trimming the opening spell (`ΔmacroF1 = -0.0004` to `-0.0040`).
  - **Crucial insight & decision outcome**: Because the scorer's `_forward_fill_x()` literally just forward-fills the previous predicted label for `spellF1` while penalising `macroF1` on every `X` frame, **emitting `X` on graded frames is strictly dominated by forward-filling the previous label ourselves in code** (which gets the exact same `spellF1` with higher `macroF1`). Therefore: **emit `X` on all `40,462` ungraded frames (`0..11,184`, `90,652..110,807`, `190,198..199,318`), and emit concrete `H`/`A`/`D` labels on all `158,857` graded frames.**

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Precision / Recall by boundary margin**:
    - **Strict OOB (`< 0` or `> 1`, `106` frames)**: **`98.11%` instant precision** for `D` (`104/106` frames; `0` `H`, `2` `A`), **`100.0%` precision within a `+5s (125f)` horizon**, **`10.67%` recall on ball-present `D`**, and **`7.08%` recall on all `D` frames**.
    - **Inner margin `0.005` (`< 0.005` or `> 0.995`, `230` frames)**: instant precision drops to **`62.61%`** (`+5s` precision `69.57%`), recall on present `D` rises to `14.77%`.
    - **Inner margin `0.010` (`334` frames)**: instant precision **`45.81%`**; **Inner margin `0.020` (`590` frames)**: instant precision **`27.80%`**.
  - **Lead / lag against the 4 `D` runs**: Strict OOB fires on **2 of the 4 `D` runs**—crossing the line **`2 frames (-0.08s)` before** `D` Run #4 (`18,027..18,389`, fire at `18,025`) and **`76 frames (+3.04s)` after** the start of `D` Run #3 (`17,380..17,838`, fire at `17,456`). The other 2 `D` runs (fouls / in-pitch stoppages) never go OOB.
  - **Decision outcome**: Strict OOB (`< 0` or `> 1`) is a near-100%-precision dead-ball trigger, whereas inner margins (`≥ 0.5%` inside the line) are heavily polluted by live wing play. Use strict OOB + signed distance outside `[0, 1]` as a high-precision `D` feature/trigger.

### 6.2 Q6.2 — Do players slow down during dead balls? `[GT-5MIN]`

- **Metric / Measurement**: mean and p90 of player speeds (`p*_speed`, in m/s) per frame — averaged
  over active players and split by team — plus ball speed, conditioned on `H` / `A` / `D`. Also the
  frame-level separability of the best single threshold.
- **Why it matters / what it gates**: fouls, injuries and set-piece set-up produce a collective
  velocity drop. If it separates cleanly it is a cheap, ball-free dead-ball feature — valuable
  precisely on the frames where the ball is missing.
- **Decision rule**: a feature is worth building only if its best single-threshold separation is
  clearly above the `D` base rate (19.6% of frames).
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Speed distributions conditioned on GT class**:

    | GT Class | Frames | Mean Player Speed (m/s) | Median Player Speed (m/s) | Mean `p90` Player Speed (m/s) | Home Mean Speed (m/s) | Away Mean Speed (m/s) | Derived Ball Speed Med (m/s) |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | `H` | 3,454 | **2.21** | 2.13 | 3.43 | 2.11 | 2.31 | 5.30 |
    | `A` | 2,577 | **2.61** | 2.52 | 4.01 | 2.58 | 2.64 | 6.56 |
    | `D` | 1,469 | **1.42** | 1.19 | **2.26** | 1.38 | 1.46 | **2.70** |

  - **Single-threshold separability**: The single rule `mean_player_speed < 1.40 m/s` achieves **`Precision = 59.9%`**, **`Recall = 76.8%`**, and **`F1 = 0.6728`** for class `D` (over **3.4× the `19.59%` base rate**, using **zero ball information**).
  - **Decision outcome**: Collective player speed (`mean`, `p90`, fraction of walking players `< 1.5 m/s`, rolling 2s–5s speed) is our strongest ball-independent dead-ball feature and is essential for separating `D` from occluded live play (`A`/`H`) when the ball is `NaN`.

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
- **Answer / Findings (`[FULL]`)**:
  - **Coverage**: `139,587 / 199,319` video frames (`70.03%`) and **`130,698 / 158,857` graded frames (`82.27%`)** are covered by `≥ 1` Smart Tagging interval.
  - **Concurrency histogram on `158,857` graded frames**:
    - `0` active tags: `28,159f` (**`17.73%`**)
    - `1` active tag: `38,440f` (**`24.20%`**)
    - `2` active tags: `42,005f` (**`26.44%`**)
    - `3` active tags: `20,404f` (**`12.84%`**)
    - `4` active tags: `13,816f` (**`8.70%`**)
    - `5+` active tags: `16,033f` (**`10.09%`**, up to `12` simultaneous tags)
  - **Per-code graded frame coverage (top codes)**: `PLAYERS IN THE BOX` (`72,639f / 45.73%`), `BALL IN FINAL THIRD` (`58,690f / 36.95%`), `SET PIECES` (`43,899f / 27.63%`), `PROGRESSION` (`31,118f / 19.59%`), `DEFENDING IN MIDDLE THIRD` (`31,118f / 19.59%`), `CREATING CHANCES` (`22,500f / 14.16%`), `DEFENDING IN DEFENSIVE THIRD` (`22,500f / 14.16%`), `BUILD UP` (`19,021f / 11.97%`), `DEFENDING IN ATTACKING THIRD` (`19,021f / 11.97%`), `BALL IN THE BOX` (`15,912f / 10.02%`), `SHORT PASS` (`12,068f / 7.60%`), `HIGH PASS` (`8,735f / 5.50%`), `ATTACKING TRANSITION` & `DEFENSIVE TRANSITION` (`8,354f / 5.26%` each), `LONG BALL` (`4,867f / 3.06%`).

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Boundary offsets for the 3 window-overlapping `SET PIECES` (`875f / 35s` or `125f / 5s` templates)**:
    1. `Kick Off (11,097..11,222, Team=N/A)`: Start offset **`-88f`** to `11,185` (`D → alive` start); End offset **`+37f`** from `11,185`.
    2. `Free Kick (17,533..18,408, Athletic Club)`: Start offset **`-305f`** to `17,838` (`D → A` restart); End offset **`+19f`** from `18,389` (`D → H` restart, and `+381f` from `18,027` `A → D`).
    3. `Goal Kick (18,320..19,195, Barcelona)`: Start offset **`-69f`** to `18,389` (`D → H` restart); End offset **`+806f`** from `18,389`.
  - **Decision outcome**: `0/3` starts and `0/3` ends align within `±50f` of their target transitions (`SET PIECES` boxes are coarse fixed-width `35s` / `875f` video-clip windows around the restart event, typically starting `~70–305f` before the restart). Per the decision rule (`< 50%` boundary agreement at `±50f`), **do not use `SET PIECES` boundaries as transition triggers**, though the `SET PIECES` team tag (when non-`N/A`, e.g. Goal Kick / Corner / Free Kick) carries `96.2%` alive-frame team agreement (see Q7.3).

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
- **Answer / Findings (`[GT-5MIN]`)**:
  - **Complete table across all 13 codes active in the window (both All-Window frames and Alive-Only `H/A` frames)**:

    | Code | Role | Window Frames (Alive Frames) | Raw `Tag == GT` (All) | Implied `Poss == GT` (All) | Implied `Poss == GT` (**Alive `H/A` Only**) | Decision (`≥60%` Alive) |
    |---|---|---:|---:|---:|---:|---|
    | `BUILD UP` | Offensive (`+1`) | 325 (295) | 90.77% | **90.77%** | **100.00%** | **KEEP** |
    | `DEFENDING IN ATTACKING THIRD` | Defensive (`-1`) | 325 (295) | 0.00% | **90.77%** | **100.00%** | **KEEP** |
    | `LONG BALL` | Offensive (`+1`) | 204 (204) | 90.20% | **90.20%** | **90.20%** | **KEEP** |
    | `PROGRESSION` | Offensive (`+1`) | 2,707 (2,707) | 72.59% | **72.59%** | **72.59%** | **KEEP** |
    | `DEFENDING IN MIDDLE THIRD` | Defensive (`-1`) | 2,707 (2,707) | 27.41% | **72.59%** | **72.59%** | **KEEP** |
    | `ATTACKING TRANSITION` | Offensive (`+1`) | 277 (277) | 67.15% | **67.15%** | **67.15%** | **KEEP** |
    | `DEFENSIVE TRANSITION` | Defensive (`-1`) | 277 (277) | 32.85% | **67.15%** | **67.15%** | **KEEP** |
    | `SET PIECES` (tagged team) | Offensive (`+1`) | 1,431 (576) | 39.13% | 39.13% | **96.18%** | **KEEP (for H/A)** |
    | `CREATING CHANCES` | Offensive (`+1`) | 675 (170) | 25.19% | 25.19% | **100.00%** | **KEEP (for H/A)** |
    | `DEFENDING IN DEFENSIVE THIRD` | Defensive (`-1`) | 675 (170) | 0.00% | 25.19% | **100.00%** | **KEEP (for H/A)** |
    | `BALL IN FINAL THIRD` | Offensive (`+1`) | 3,886 (3,039) | 42.10% | 42.10% | **53.70%** | **DROP (`<60%`)** |
    | `PLAYERS IN THE BOX` | Offensive (`+1`) | 4,764 (3,353) | 32.33% | 32.33% | **45.93%** | **DROP (`<60%`)** |
    | `BALL IN THE BOX` | Offensive (`+1`) | 786 (552) | 30.92% | 30.92% | **44.02%** | **DROP (`<60%`)** |

  - **Key insight**: Notice why `CREATING CHANCES`, `DEFENDING IN DEFENSIVE THIRD`, and `SET PIECES` had low all-frame agreement (`25.2%` and `39.1%`): their intervals in the window overlapped long `D` stoppages (`505/675` frames of `CREATING CHANCES` were `D`), **not** opposing-team possession! Conditional on the ball being alive (`H` or `A`), `CREATING CHANCES` and `DEFENDING IN DEFENSIVE THIRD` have **100.0% implied-possessor accuracy** and `SET PIECES` has **96.2%**. Conversely, spatial zone tags (`BALL IN FINAL THIRD`, `PLAYERS IN THE BOX`, `BALL IN THE BOX`) remain `< 54%` even on alive frames and are dropped as team priors.

### 7.4 Q7.4 — How much do we lose to `Team = N/A`? `[FULL]`

- **Metric / Measurement**: count of instances with `Team = N/A` by code, and the frame coverage those
  instances represent; specifically confirm whether **all 26 throw-ins** are `N/A` as documented.
- **Why it matters / what it gates**: DATA.md warns that every throw-in is untagged, "exactly the
  restarts where knowing the side would be worth the most". If that holds, the team prior has a
  structural hole exactly at `D → alive` boundaries — the highest-value frames in the whole problem.
- **Decision rule**: if throw-ins are all `N/A`, do not attempt to fill them from tags; resolve them
  from the tracking's last-touch geometry instead, and record the limitation in the technical document.
- **Answer / Findings (`[FULL]`)**:
  - **Complete `Team = N/A` breakdown (`36` instances, `26,500` total instance-frames)**:
    - `SET PIECES / Throw-in`: **`26 / 26` instances (`100.0%`)**, `22,750` frames (`364` frames in window).
    - `SET PIECES / Kick Off (After Goal)`: **`4 / 4` instances (`100.0%`)**, `500` frames.
    - `SET PIECES / Kick Off (Start)`: **`2 / 2` instances (`100.0%`)**, `250` frames (`38` frames in window).
    - `CREATING CHANCES / None`: `2 / 30` instances (`6.67%`), `1,500` frames.
    - `DEFENDING IN DEFENSIVE THIRD / None`: `2 / 30` instances (`6.67%`), `1,500` frames.
  - Confirmed: **100% of throw-ins (`26/26`) and kick-offs (`6/6`) have `Team = N/A`**, so throw-in and kick-off restart attribution must come from player-ball tracking geometry.

---

## 8. Coverage and geometry checks

### 8.1 Q8.1 — Is player tracking density stable? `[FULL]`

- **Metric / Measurement**: distribution of non-`NaN` tracked slots per frame, by match phase (pre-match,
  1st half, halftime, 2nd half, post-match); share of frames with < 10 or > 22 tracked objects.
- **Why it matters / what it gates**: collective features (team centroid, pressing line, compactness)
  are biased by missing players, and the bias is not uniform across the match.
- **Measured (2026-10-31)**: mean 20.5 over graded frames (DATA.md); confirm here.
- **Answer / Findings (`[FULL]`)**:
  - **Phase-by-phase player tracking density & homography quality (Q8.1 + Q8.3)**:

    | Match Phase | Frames | Mean Tracked Players | `p10 / Med / p90` Players | `< 10` Players (%) | `> 22` Players (%) | Reproj Err Med / p95 |
    |---|---:|---:|---|---:|---:|---|
    | `1. Pre-match (0–11,184)` | 11,185 | 15.83 | `0 / 22 / 23` | 26.86% | 26.27% | `0.340 / 0.568` |
    | `2. 1st Half (11,185–90,651)` | 79,467 | **20.12** | `16 / 21 / 22` | **4.18%** | **0.87%** | `0.224 / 0.352` |
    | `3. Halftime (90,652–110,807)` | 20,156 | 12.51 | `6 / 12 / 19` | 31.61% | 2.13% | `0.395 / 4.830` |
    | `4. 2nd Half (110,808–190,197)` | 79,390 | **20.85** | `19 / 22 / 22` | **1.91%** | **3.85%** | `0.237 / 0.363` |
    | `5. Post-match (190,198–199,318)` | 9,121 | 15.10 | `7 / 14 / 23` | 24.48% | 0.25% | `0.340 / 0.659` |

  - Combined graded play (`158,857` frames) averages **`20.49` tracked players** (`3.05%` of frames have `< 10` players, `2.36%` have `> 22`). Team-aggregate spatial features should normalize by active player count per team (`n_H_active`, `n_A_active`) and pass active counts as reliability features.

### 8.2 Q8.2 — Does the coordinate frame flip at halftime? `[FULL]`

- **Metric / Measurement**: team-mean `x` position per half, and the sign of the change; test whether
  normalising by attack direction is required before any "own half / final third" feature.
- **Why it matters / what it gates**: teams switch ends. A feature like "defensive third" silently
  inverts for one team in one half if the frame is not attack-direction-normalised. One cell to check,
  expensive to find later.
- **Decision rule**: if the team-mean `x` ordering reverses across halves, every spatial feature must be
  expressed in attack-relative coordinates.
- **Answer / Findings (`[FULL]`)**:
  - **1st Half (`11,185–90,651`)**: Home `mean_x = 0.4659`, Away `mean_x = 0.5832`, difference `Home − Away = -0.1173` (**Home attacks Left → Right `+x`**; Away attacks Right → Left `-x`).
  - **2nd Half (`110,808–190,197`)**: Home `mean_x = 0.5300`, Away `mean_x = 0.4151`, difference `Home − Away = +0.1149` (**Home attacks Right → Left `-x`**; Away attacks Left → Right `+x`).
  - **Decision outcome**: The pitch `x`-axis is fixed to the stadium camera and **reverses team orientation at halftime**. All longitudinal (`x`) features and velocity `vx` components must be attack-direction-normalized (`x_norm = 1 - x` in 2nd Half, or signed relative to each team's defending goal).

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
