problem="hero"
stamp="20230515-094404"
step=1000000
testdata="data0511a"
savedir="analysis0511a-v4-ind"

# you can comment out "--save_raw --save_prompt --batch_size 1 \" to speed up
# but you will not be able to plot the raw data then
# if you keep it, for each demo_num_begin, it takes about 90 seconds to finish

for demo_num_begin in 1 2 3 4 5
do
  demo_num_end=$(expr $demo_num_begin + 1)
  python3 analysis_accelerate.py --analysis_dir $savedir --task ind \
                                  --problem $problem --stamp $stamp --step $step \
                                  --test_data_dirs $testdata --test_data_globs "test*" \
                                  --k_dim 3 --k_mode itx \
                                  --demo_num_begin $demo_num_begin --demo_num_end $demo_num_end \
                                  --save_raw --save_prompt --batch_size 1 \
                                  >out-analysis-ind-$stamp-$step-$testdata-$demo_num_begin-$demo_num_end.log 2>&1
done

echo "Done"



