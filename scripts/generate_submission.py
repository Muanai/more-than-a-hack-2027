"""Deterministic CLI entrypoint to regenerate the competition submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_match_data  # noqa: E402
from src.pipelines.ablation import run_ablation4_step1_physics  # noqa: E402
from src.pipelines.baseline import save_submission  # noqa: E402
from src.pipelines.step2_learned import (  # noqa: E402
    run_ablation5_learned_viterbi,
    run_ablation6_learned_hsmm,
)
from src.pipelines.step3_em import (  # noqa: E402
    run_ablation7_step3_viterbi,
    run_ablation8_step3_hsmm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate More than a Hack 2027 Phase 1 submission CSV.")
    parser.add_argument(
        "--out",
        default="submissions/sendiri aj.csv",
        help="Output CSV path (default: 'submissions/sendiri aj.csv')",
    )
    parser.add_argument(
        "--model",
        choices=("step3_hsmm", "step3_viterbi", "step2_hsmm", "step2_viterbi", "step1_physics"),
        default="step3_hsmm",
        help="Pipeline stage to execute (default: step3_hsmm)",
    )
    args = parser.parse_args()

    data = load_match_data()
    runners = {
        "step3_hsmm": run_ablation8_step3_hsmm,
        "step3_viterbi": run_ablation7_step3_viterbi,
        "step2_hsmm": run_ablation6_learned_hsmm,
        "step2_viterbi": run_ablation5_learned_viterbi,
        "step1_physics": run_ablation4_step1_physics,
    }
    res = runners[args.model](data=data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_submission(
        res.labels,
        out_path=out_path,
        ungraded_to_x=True,
        graded_mask=data.graded_mask,
    )
    print(f"Generated {out_path} ({len(data.frames):,} rows) using {args.model}.")


if __name__ == "__main__":
    main()
