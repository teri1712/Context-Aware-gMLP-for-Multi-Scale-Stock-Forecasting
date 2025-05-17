market_name=("NASDAQ" "SP500" "crypto")
stock_num=("1026" "474" "117")
valid_index=("756" "1006" "620")
test_index=("1008" "1259" "827")

baseline_market_values=("20" "8" "10")

market_values=("48" "15" "25")
depth_values=("3" "3" "5")

echo "Result for baseline"
./run_original.sh

echo "Result"
./run_extended.sh
