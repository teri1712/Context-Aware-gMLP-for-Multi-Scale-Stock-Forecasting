import subprocess
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    print("Result for baseline")
    subprocess.run(
        ["python3", str(script_dir / "run_original.py")],
        check=True,
    )

    print("\nResult")
    subprocess.run(
        ["python3", str(script_dir / "run_extended.py")],
        check=True,
    )


if __name__ == "__main__":
    main()
