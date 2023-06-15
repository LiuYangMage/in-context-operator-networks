savedir="../analysis/analysis0521-ood"
suffix="_light" # for light dataset
# suffix="" # for heavy dataset
figs=10
testdatas=(
  "data0520_ood_odeconst$suffix"
  "data0520_ood_odelinear1$suffix"
  "data0520_ood_odelinear2$suffix"
  "data0520_ood_pdeporous_randbdry$suffix"
  "data0520_ood_seriesdamped$suffix"
  "data0520_nt_odelinear3$suffix"
  "data0520_nt_ot_rho1param$suffix"
  )

### =============================This is for k_mode = 'itx'======================

problems=(
          "hero"
          "group-3-itx" "group-3-itx" "group-3-itx" "group-3-itx"
          )
stamps=(
        "20230515-094404" # hero v4
        "20230522-234219"  "20230524-165312"  "20230525-010119"  "20230525-090937" 
        )
step=(
      1000000
      200000 200000 200000 200000
      )

for index in ${!problems[*]}
do 
  echo "index: $index" "problem: ${problems[$index]}" "stamp: ${stamps[$index]}" "step: ${step[$index]}"
  problem=${problems[$index]}
  stamp=${stamps[$index]}
  step=${step[$index]}
  for testdata in "${testdatas[@]}"
  do
    for demo_num_begin in 5
    do
      echo "testdata: $testdata" "demo_num_begin: $demo_num_begin"
      demo_num_end=$(expr $demo_num_begin + 1)
      python3 analysis_accelerate.py --analysis_dir $savedir --task ood  --figs $figs\
                                    --problem $problem --stamp $stamp --step $step \
                                    --test_data_dirs $testdata --test_data_globs "test*" \
                                    --demo_num_begin $demo_num_begin --demo_num_end $demo_num_end \
                                    --test_config_filename "test_config_ood.json" \
                                    --k_dim 3 --k_mode itx \
                                    >out-analysis-ood-$problem-$stamp-$step-$testdata-$demo_num_begin-$demo_num_end.log 2>&1
    done
  done
done


## =============================fake operator ======================

testdatas=("data0520_nt_odelinear3$suffix")

for testdata in "${testdatas[@]}"
do
  for mode in "real_op" "fake_op"
  do
    echo "testdata: $testdata" "mode: $mode"
    python3 analysis_accelerate.py --analysis_dir $savedir --task ood --mode $mode --figs $figs\
                                  --test_data_dirs $testdata --test_data_globs "test*" \
                                  --demo_num_begin 5 --demo_num_end 6 \
                                  --test_config_filename "test_config_ood.json" \
                                  >out-analysis-ood-$mode-$testdata.log 2>&1
  done
done

## =============================fake demo======================

testdatas=("data0520_nt_odelinear3$suffix")

problems=(
          "hero"
          "group-3-itx" 
          )
stamps=(
        "20230515-094404" # hero v4
        "20230522-234219" 
        )
step=(
      1000000
      200000
      )

for testdata in "${testdatas[@]}"
do
  for index in ${!problems[*]}
  do 
    echo "fake demo, index: $index" "problem: ${problems[$index]}" "stamp: ${stamps[$index]}" "step: ${step[$index]}"
    problem=${problems[$index]}
    stamp=${stamps[$index]}
    step=${step[$index]}
    for demo_num_begin in 5
    do
      echo "testdata: $testdata" "demo_num_begin: $demo_num_begin"
      demo_num_end=$(expr $demo_num_begin + 1)
      python3 analysis_accelerate.py --analysis_dir $savedir --task ood --mode fake_demo --batch_size 50  --figs $figs\
                                    --problem $problem --stamp $stamp --step $step \
                                    --test_data_dirs $testdata --test_data_globs "test*" \
                                    --demo_num_begin $demo_num_begin --demo_num_end $demo_num_end \
                                    --test_config_filename "test_config_ood.json" \
                                    --k_dim 3 --k_mode itx \
                                    >out-analysis-ood-$problem-$stamp-$step-fake_demo-$testdata-$demo_num_begin-$demo_num_end.log 2>&1
    done
  done
done

echo "Done"
