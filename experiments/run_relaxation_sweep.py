"""
Master script to run the complete relaxation sweep experiment.

This script executes the full pipeline:
1. Generate datasets with varying basin strengths
2. Train sigmoid AANN on each dataset
3. Evaluate pattern completion for each model
4. Generate plots and summary

Run this to reproduce the "smooth attractor basins enable learning" result.
"""

import subprocess
from pathlib import Path


def run_script(script_path: str, description: str):
    """Run a Python script and report status."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)

    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ Error running {script_path}")
        return False

    print(f"\n✓ Completed: {description}")
    return True


def main():
    """Run the complete pipeline."""

    print("\n" + "=" * 70)
    print("RELAXATION SWEEP EXPERIMENT")
    print("Testing: Smooth Attractor Basins Enable SGD Learning")
    print("=" * 70)

    # define pipeline steps
    steps = [
        (
            "experiments/generate_relaxation_sweep.py",
            "Generate datasets with varying basin strengths [0.1, 0.3, 0.5, 0.7, 0.9]"
        ),
        (
            "experiments/train_relaxation_sweep.py",
            "Train sigmoid AANN models on each dataset"
        ),
        (
            "experiments/eval_relaxation_sweep.py",
            "Evaluate pattern completion and generate plots"
        ),
    ]

    # run pipeline
    for script_path, description in steps:
        success = run_script(script_path, description)
        if not success:
            print("\n" + "=" * 70)
            print("❌ Pipeline failed. Please check errors above.")
            print("=" * 70)
            return

    # print summary
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)

    output_dir = Path("experiments/results/relaxation_sweep_eval")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - eval_results.json: numerical results")
    print(f"  - relaxation_sweep_results.png: plot showing phase accuracy vs basin strength")
    print(f"  - relaxation_sweep_results.pdf: publication-quality plot")

    print("\n" + "=" * 70)
    print("KEY FINDING:")
    print("Phase recovery accuracy should increase with basin strength,")
    print("demonstrating that smooth attractor basins enable SGD learning.")
    print("=" * 70)


if __name__ == "__main__":
    main()
