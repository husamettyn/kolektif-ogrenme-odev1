#!/usr/bin/env python3
"""Generate objective function plots without trajectories."""
import argparse
from pathlib import Path

from plotting.plot_results import plot_objective
from objectives.registry import get_objective, OBJECTIVES


def main():
    parser = argparse.ArgumentParser(description="Generate objective function plots")
    parser.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        default=list(OBJECTIVES.keys()),
        choices=list(OBJECTIVES.keys()),
        help="Objectives to plot (default: all)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="comp/objectives",
        help="Output directory for objective plots",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[OBJECTIVES] Generating plots in {outdir}...")
    for obj_name in args.objectives:
        obj_fn, obj_grad, bounds, start_default, global_mins = get_objective(obj_name)
        obj_img_path = outdir / f"{obj_name}.png"
        plot_objective(
            bounds=bounds,
            objective_fn=obj_fn,
            objective_label=obj_name,
            outpath=str(obj_img_path),
        )
        print(f"  [OK ] {obj_name} -> {obj_img_path}")
    print(f"[OBJECTIVES] Done. Generated {len(args.objectives)} plots.")


if __name__ == "__main__":
    main()
