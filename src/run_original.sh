market_name=("NASDAQ" "SP500" "crypto")
stock_num=("1026" "474" "117")
valid_index=("756" "1006" "620")
test_index=("1008" "1259" "827")

market_values=("20" "8" "10")

cd original/
for i in "${!market_name[@]}"; do
  if [ "${market_name[$i]}" == "SP500" ]; then
    git checkout b657771600cd8b0c40267ece85412e2a21aafc17
    python3 train.py ${market_name[$i]} ${stock_num[$i]} ${valid_index[$i]} ${test_index[$i]} ${market_values[$i]}
    git checkout market-context-combined-with-gating
  else
    python3 train.py ${market_name[$i]} ${stock_num[$i]} ${valid_index[$i]} ${test_index[$i]} ${market_values[$i]}
  fi
done