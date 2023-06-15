problem="hero"
stamp="20230515-094404"
step=1000000
testdata="data0511a"
savedir="analysis0511a-v4-len"


len_demo_cond_lens=(10 20 30 40 50 60 80 100 200 500)
len_demo_qoi_lens=(10 20 30 40 50 60 80 100 200 500)
len_quest_cond_lens=(10 20 30 40 50 60 80 100 200 500)

len_quest_qoi_len=2600

for index in ${!len_demo_cond_lens[*]}
do 
  for demo_num_begin in 5 4 3 2 1
  do
    demo_num_end=$(expr $demo_num_begin + 1)
    len_demo_cond_len=${len_demo_cond_lens[$index]}
    len_demo_qoi_len=${len_demo_qoi_lens[$index]}
    len_quest_cond_len=${len_quest_cond_lens[$index]}
    echo "len_demo_cond_len: $len_demo_cond_len" "len_demo_qoi_len: $len_demo_qoi_len" "len_quest_cond_len: $len_quest_cond_len" "len_quest_qoi_len: $len_quest_qoi_len" \
         "demo_num_begin" $demo_num_begin "demo_num_end" $demo_num_end

    python3 analysis_accelerate.py --analysis_dir $savedir-$len_demo_cond_len-$len_demo_qoi_len-$len_quest_cond_len-$len_quest_qoi_len \
                                  --task len --problem $problem --stamp $stamp --step $step \
                                  --test_data_dirs $testdata --test_data_globs "test_mfc_gparam_hj_forward22*" \
                                  --demo_num_begin $demo_num_begin --demo_num_end $demo_num_end \
                                  --test_config_filename "test_config_len.json" --cond_len 2400 --qoi_len 2600 \
                                  --len_demo_cond_len $len_demo_cond_len --len_demo_qoi_len $len_demo_qoi_len \
                                  --len_quest_cond_len $len_quest_cond_len --len_quest_qoi_len $len_quest_qoi_len \
                                  --batch_size 1 --save_raw --save_prompt --figs 10 \
                                  --k_dim 3 --k_mode itx \
                                  >out-analysis-len-$len_demo_cond_len-$len_demo_qoi_len-$len_quest_cond_len-$len_quest_qoi_len-$stamp-$step-$testdata-$demo_num_begin-$demo_num_end.log 2>&1
  done
done

echo "Done"



