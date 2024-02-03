gpu=0

icon_stamp=icon_lm_learn_20231005-094726-pde3-inverse
tune_stamp=icon_lm_fno_20240121-203841-pde3-inverse
restore_dir=/home/shared/icon/save/user/ckpts/fno_pretrain/20240121-203841
model_config=model_fno_pde_config.json

CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name fno --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range   0,100 >tune-$tune_stamp-0-100.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name fno --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 100,200 >tune-$tune_stamp-100-200.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name fno --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 200,300 >tune-$tune_stamp-200-300.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name fno --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 300,400 >tune-$tune_stamp-300-400.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu python3 finetune.py --model_name fno --model_config $model_config --icon_stamp $icon_stamp --tune_stamp $tune_stamp --restore_dir $restore_dir --tune_bid_range 400,500 >tune-$tune_stamp-400-500.log 2>&1 &


echo "Done"



