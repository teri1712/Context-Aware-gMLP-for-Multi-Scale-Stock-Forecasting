market_name=("NASDAQ" "SP500" "crypto")
stock_num=("1026" "474" "117")
valid_index=("756" "1006" "620")
test_index=("1008" "1259" "827")

market_values=("20" "8" "10")

cd original/
for i in "${!market_name[@]}"; do
  python3 train.py ${market_name[$i]} ${stock_num[$i]} ${valid_index[$i]} ${test_index[$i]} ${market_values[$i]}
done
