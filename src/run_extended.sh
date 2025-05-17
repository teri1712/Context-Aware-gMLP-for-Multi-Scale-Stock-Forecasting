market_name=("NASDAQ" "SP500" "crypto")
stock_num=("1026" "474" "117")
valid_index=("756" "1006" "620")
test_index=("1008" "1259" "827")

market_values=("48" "15" "25")
depth_values=("3" "3" "5")

cd extended/
for i in "${!market_name[@]}"; do
  for j in "${!stock_num[@]}"; do
    for k in "${!valid_index[@]}"; do
      for l in "${!test_index[@]}"; do
        for x in "${!market_values[@]}"; do
          for y in "${!depth_values[@]}"; do
            python3 train.py ${market_name[$i]} ${stock_num[$j]} ${valid_index[$k]} ${test_index[$l]} ${market_values[$x]} ${depth_values[$y]}
          done
        done
      done
    done
  done
done

