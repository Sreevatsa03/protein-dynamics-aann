import subprocess
from pathlib import Path


def run_script(script_path: str, description: str):
    """Run a Python script and report status."""
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70)

    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"\nError running {script_path}")
        return False

    print(f"\nCompleted: {description}")
    return True


def main():
    print("\n" + "=" * 70)
    print("Relaxation Sweep Pipeline")
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
            print("Pipeline failed. Please check errors above.")
            print("=" * 70)
            return

    # print summary
    print("Relaxation sweep completed.")

    output_dir = Path("experiments/results/relaxation_sweep_eval")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
