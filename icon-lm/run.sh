#  Encoder-decoder ICON (Single-Modal):
CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'icon' --epochs 100 \
  --train_batch_size 32 --train_data_dirs '/home/shared/icon/data/data0910c' \
  --model_config_filename 'model_icon_config.json' \
  --train_config_filename 'train_icon_config.json' \
  --test_config_filename 'test_icon_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon --loss_mode nocap \
  --nodeterministic --seed 1 --vistest  --tfboard


# ICON-LM (Single-Modal):
CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'icon_lm' --epochs 100 \
  --train_batch_size 24 --train_data_dirs '/home/shared/icon/data/data0910c' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
  --nodeterministic --seed 1 --vistest --tfboard 


#GPT-2 (Multi-Modal):
#Add `--unpretrained` option to start from an unpretrained GPT-2 model.
python3 run.py --problem 'icon_gpt2' --epochs 100 \
  --train_batch_size 10 --train_data_dirs '/home/shared/icon/data/data0910c' \
  --model_config_filename 'model_gpt2_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 0,1,3,5 --test_caption_id_list -1,0 \
  --model gpt2 --backend torch --loss_mode nocap,cap \
  --nodeterministic --trainable_mode all --seed 1 --vistest --tfboard 

# Pretrain DeepONet on Problem #14:
CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'deepo_pretrain' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/home/shared/icon/data/data0910c' \
  --train_data_globs 'train_pde_cubic_spatial_inverse*' \
  --test_data_globs 'test_pde_cubic_spatial_inverse*' \
  --model_config_filename 'model_deepo_pde_config.json' \
  --train_config_filename 'train_lm_pde_full_config.json' \
  --test_config_filename 'test_lm_pde_full_config.json' \
  --test_demo_num_list 1,3,5 --model deepo \
  --backend torch --loss_mode demo_quest \
  --nodeterministic --trainable_mode all --seed 1 --vistest --tfboard

# Pretrain FNO on Problem #14:
CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'fno_pretrain' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/home/shared/icon/data/data0910c' \
  --train_data_globs 'train_pde_cubic_spatial_inverse*' \
  --test_data_globs 'test_pde_cubic_spatial_inverse*' \
  --model_config_filename 'model_fno_pde_config.json' \
  --train_config_filename 'train_lm_pde_full_config.json' \
  --test_config_filename 'test_lm_pde_full_config.json' \
  --test_demo_num_list 1,3,5 --model fno \
  --backend torch --loss_mode demo_quest \
  --nodeterministic --trainable_mode all --seed 1 --vistest --tfboard 
