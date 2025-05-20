from pathlib import Path
from subprocess import run
from typing import List

SP500_OPTIMAL_SCALE = 'b657771600cd8b0c40267ece85412e2a21aafc17'


def main() -> None:
    # Parameters
    market_name: List[str] = ["SP500", "crypto"]
    stock_num: List[str] = ["474", "117"]
    valid_index: List[str] = ["1006", "620"]
    test_index: List[str] = ["1259", "827"]
    market_values: List[str] = ["8", "10"]

    original_dir = Path(__file__).resolve().parent / "original"
    for i in range(len(market_name)):

        if market_name[i] == "SP500":
            run(["git", "checkout", SP500_OPTIMAL_SCALE], check=True)

        cmd = [
            "python3",
            "train.py",
            market_name[i],
            stock_num[i],
            valid_index[i],
            test_index[i],
            market_values[i],
        ]

        run(cmd, check=True, cwd=original_dir)

        if market_name[i] == "SP500":
            run(["git", "checkout", "HEAD"], check=True)


if __name__ == "__main__":
    main()
