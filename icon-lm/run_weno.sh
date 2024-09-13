#  Conservation Law L2 loss with batch size 8 for forward and 8 for reverse
# set "--train_batch_size 2n" for batch size n for forward and n for reverse 
python3 run.py --problem 'icon_weno' --epochs 100 --train_batch_size 16 \
  --train_data_dirs '/home/shared/icon/data/data0904_weno_cubic' \
  --test_data_dirs '/home/shared/icon/data/data0904_weno_cubic' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_weno_config.json' \
  --test_config_filename 'test_weno_config.json' \
  --test_demo_num_list 0,1,3,5 --model icon_lm --loss_mode nocap \
  --vistest --nodeterministic --tfboard



# Conservation Law with batch size 4 for forward and 4 for consistency loss
python3 run.py --problem 'icon_weno' --epochs 100 --train_batch_size 4 \
  --train_data_dirs '/home/shared/icon/data/data0904_weno_cubic' \
  --test_data_dirs '/home/shared/icon/data/data0904_weno_cubic' \
  --train_data_globs 'train*forward*' --test_data_globs 'test*forward*' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_weno_config.json' \
  --test_config_filename 'test_lm_weno_config.json' \
  --test_demo_num_list 0,1,3,5 --model icon_lm --loss_mode consist \
  --vistest --nodeterministic --tfboard


# Pretrain DeepONet for f = 0.2 u^3 + 0.2 u^2 + 0.2 u
CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'weno_deepo_pretrain' --epochs 10 --train_batch_size 32 \
  --train_data_dirs '/home/shared/icon/data/data0604_weno_cubic_fix_0.2_0.2_0.2' \
  --test_data_dirs '/home/shared/icon/data/data0604_weno_cubic_fix_0.2_0.2_0.2' \
  --train_data_globs 'train*forward*' \
  --test_data_globs 'test*forward*' \
  --model_config_filename 'model_deepo_weno_config.json' \
  --train_config_filename 'train_lm_weno_config.json' \
  --test_config_filename 'test_lm_weno_config.json' \
  --test_demo_num_list 0,1,3,5 \
  --test_caption_id_list -1 \
  --model deepo --backend torch --loss_mode demo_quest \
  --nodeterministic --trainable_mode all --seed 1 --vistest --tfboard


# Pretrain FNO for f = 0.2 u^3 + 0.2 u^2 + 0.2 u
CUDA_VISIBLE_DEVICES=1 python3 run.py --problem 'weno_fno_pretrain' --epochs 10 --train_batch_size 32 \
  --train_data_dirs '/home/shared/icon/data/data0604_weno_cubic_fix_0.2_0.2_0.2' \
  --test_data_dirs '/home/shared/icon/data/data0604_weno_cubic_fix_0.2_0.2_0.2' \
  --train_data_globs 'train*forward*' \
  --test_data_globs 'test*forward*' \
  --model_config_filename 'model_fno_weno_config.json' \
  --train_config_filename 'train_lm_weno_config.json' \
  --test_config_filename 'test_lm_weno_config.json' \
  --test_demo_num_list 0,1,3,5 \
  --test_caption_id_list -1 \
  --model fno --backend torch --loss_mode demo_quest \
  --nodeterministic --trainable_mode all --seed 1 --vistest --tfboard 
