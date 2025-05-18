from pathlib import Path
from subprocess import run
from typing import List


def main() -> None:
    market_name: List[str] = ["NASDAQ", "SP500", "crypto"]
    stock_num: List[str] = ["1026", "474", "117"]
    valid_index: List[str] = ["756", "1006", "620"]
    test_index: List[str] = ["1008", "1259", "827"]
    market_values: List[str] = ["15", "32", "20"]
    depth_values: List[str] = ["3", "5", "5"]

    extended_dir = Path(__file__).resolve().parent / "extended"

    for i in range(len(market_name)):
        for j in range(len(stock_num)):
            for k in range(len(valid_index)):
                for l in range(len(test_index)):
                    for x in range(len(market_values)):
                        for y in range(len(depth_values)):
                            cmd = [
                                "python3",
                                "train.py",
                                market_name[i],
                                stock_num[j],
                                valid_index[k],
                                test_index[l],
                                market_values[x],
                                depth_values[y],
                            ]
                            run(cmd, check=True, cwd=extended_dir)


if __name__ == "__main__":
    main()
