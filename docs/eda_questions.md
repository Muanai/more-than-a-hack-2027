# Exploratory Data Analysis (EDA) — Research Questions & Findings

This document outlines the core questions that our EDA notebook must answer. Answers and quantitative findings will be populated as we run the analysis on the match assets and the 5-minute public ground truth.

---

## 1. Ground Truth & Possession Reality (Public 5-min GT)

### Q1.1: What is the label distribution during active play?
- **Metric / Measurement**: Count and percentage of frames labeled `H` (Barcelona), `A` (Athletic Club), and `D` (Dead ball) in the public 5-minute GT window (frames 11,185 to 18,684).
- **Why it matters**: Establishes class balance and baseline priors for active play.
- **Answer / Findings**: *(To be answered)*

### Q1.2: What is the duration distribution of real possessions?
- **Metric / Measurement**: Min, median, mean, and max duration of contiguous possession spells (`H` and `A`), both in frames and in seconds (25 fps).
- **Why it matters**: `spellF1` ranks entire continuous spells. Understanding the lower quartile of duration informs the minimum window size for temporal smoothing or post-processing filters.
- **Answer / Findings**: *(To be answered)*

### Q1.3: What do possession transitions look like?
- **Metric / Measurement**: Transition frequency matrix between states (`H -> A`, `H -> D`, `D -> H`, `D -> A`, `A -> H`, `A -> D`).
- **Why it matters**: Checks whether direct team-to-team turnovers (`H <-> A`) frequently occur without a dead ball (`D`) in between, or if dead-ball intervals mediate most turnovers.
- **Answer / Findings**: *(To be answered)*

---

## 2. The Missing Ball Problem & Gap Characteristics

### Q2.1: What is the distribution of missing ball gap lengths?
- **Metric / Measurement**: Frequency and duration histogram of contiguous `NaN` ball frames across the full match and specifically during active play.
- **Why it matters**: Identifies whether the missing ball is predominantly micro-flicker (1–5 frames) or prolonged tracking dropouts (multi-second).
- **Answer / Findings**: *(To be answered)*

### Q2.2: How much ball signal can be restored via short-term interpolation?
- **Metric / Measurement**: Percentage of missing ball frames that belong to short gaps ($\le 5$ frames / 0.2s, $\le 12$ frames / 0.5s, $\le 25$ frames / 1.0s).
- **Why it matters**: Determines whether a simple linear or spline interpolation can restore 20%, 50%, or 80% of lost ball positions safely.
- **Answer / Findings**: *(To be answered)*

### Q2.3: In the ground truth, what state is the match in when the ball is missing?
- **Metric / Measurement**: Confusion breakdown of ground truth labels (`H`, `A`, `D`) conditioned on `ball_is_nan == True` vs `ball_is_nan == False`.
- **Why it matters**: Tests the baseline's assumption that "missing ball == dead ball". If missing ball frequently happens during active passes/aerial duels, treating missing ball as `D` severely hurts `H` and `A` recall.
- **Answer / Findings**: *(To be answered)*

---

## 3. Player-Ball Proximity & Spatial Control

### Q3.1: What does distance-to-ball look like during active possession?
- **Metric / Measurement**: Distribution of distance from the ball to the nearest Home player ($d_H$), nearest Away player ($d_A$), and distance delta ($d_A - d_H$) conditioned on GT label (`H` vs `A`).
- **Why it matters**: Quantifies the physical boundary of "possession" in meters. Shows whether players in possession typically stay within 1.0m, 1.5m, or 2.5m.
- **Answer / Findings**: *(To be answered)*

### Q3.2: How often does the nearest player NOT belong to the team in possession?
- **Metric / Measurement**: Percentage of frames in the 5-min GT where `nearest_player_team != gt_label` (excluding `D`).
- **Why it matters**: Measures the theoretical error ceiling of pure proximity (e.g. ball in flight towards a teammate, tackling defender closer to ball).
- **Answer / Findings**: *(To be answered)*

---

## 4. Dead Ball (`D`) Signatures & Stoppage Physics

### Q4.1: Does the ball leave the pitch boundaries during dead balls?
- **Metric / Measurement**: Correlation between ball coordinates being out of bounds ($x < 0$ or $x > 1$, $y < 0$ or $y > 1$) and the `D` label.
- **Why it matters**: An out-of-bounds ball is an unambiguous indicator of a dead ball (throw-in, goal kick, corner).
- **Answer / Findings**: *(To be answered)*

### Q4.2: How do player and ball velocities behave during dead balls vs live play?
- **Metric / Measurement**: Mean and 90th percentile velocities of active players, and ball velocity, during `D` frames vs `H`/`A` frames.
- **Why it matters**: During fouls, injuries, or set piece setups, collective player velocity typically drops, giving a strong global feature for `D`.
- **Answer / Findings**: *(To be answered)*

---

## 5. Smart Tagging Coverage & Priors

### Q5.1: How much of the match timeline is covered by Smart Tagging?
- **Metric / Measurement**: Total frames and percentage of match time covered by at least one Smart Tagging interval, and breakdown by event code (`SET PIECES`, `BUILD UP`, `DEFENDING`, etc.).
- **Why it matters**: Evaluates whether Smart Tagging can be used as a dense feature or an occasional strong prior.
- **Answer / Findings**: *(To be answered)*

### Q5.2: How accurate is `SET PIECES` as a detector for Dead Ball (`D`)?
- **Metric / Measurement**: Precision and recall of `SET PIECES` tags against `D` frames in the 5-min GT.
- **Why it matters**: Tests whether set piece tag boundaries align directly with dead-ball periods and restarts.
- **Answer / Findings**: *(To be answered)*

### Q5.3: When Smart Tagging assigns a `Team`, how often does that team actually have possession?
- **Metric / Measurement**: Agreement rate between `smart_tag.team` and ground truth label (`H` for Barcelona, `A` for Athletic Club) across tagged intervals in the 5-min GT.
- **Why it matters**: Checks whether phase tags (e.g. `BUILD UP`, `ATTACKING TRANSITION`) can be trusted as direct team indicators.
- **Answer / Findings**: *(To be answered)*

---

## 6. Match Structure & Data Quality Checks

### Q6.1: What is the tracking density across all 199,319 frames?
- **Metric / Measurement**: Distribution of active (non-NaN) player tracks per frame across the entire recording.
- **Why it matters**: Confirms whether player tracking remains consistent (~20–22 players) during match halves and drops during halftime/pre-match.
- **Answer / Findings**: *(To be answered)*

### Q6.2: How do the camera homography reprojection errors look across the match?
- **Metric / Measurement**: Median and 95th percentile `reprojection_error` over 1st half, halftime, and 2nd half.
- **Why it matters**: Identifies any unstable camera segments or periods where tracking quality may degrade.
- **Answer / Findings**: *(To be answered)*
