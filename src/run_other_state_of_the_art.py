import sys
from pathlib import Path
from subprocess import run
from typing import List

SP500_OPTIMAL_SCALE = 'b657771600cd8b0c40267ece85412e2a21aafc17'

params = sys.argv[1:]


def main() -> None:
    # Parameters
    model = params[0];
    market_name: List[str] = ["NASDAQ", "SP500", "crypto"]
    stock_num: List[str] = ["1026", "474", "117"]
    valid_index: List[str] = ["756", "1006", "620"]
    test_index: List[str] = ["1008", "1259", "827"]
    market_values: List[str] = ["20", "8", "10"]

    original_dir = Path(__file__).resolve().parent / model
    for i in range(len(market_name)):
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


if __name__ == "__main__":
    main()
