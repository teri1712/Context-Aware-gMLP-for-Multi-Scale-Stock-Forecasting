market_name=("NASDAQ" "SP500" "crypto")
stock_num=("1026" "474" "117")
valid_index=("756" "1006" "620")
test_index=("1008" "1259" "827")

market_values=("15" "32" "20")
depth_values=("3" "5" "5")

cd extended/
for i in "${!market_name[@]}"; do
  python3 train.py ${market_name[$i]} ${stock_num[$i]} ${valid_index[$i]} ${test_index[$i]} ${market_values[$i]} ${depth_values[$i]}
done

