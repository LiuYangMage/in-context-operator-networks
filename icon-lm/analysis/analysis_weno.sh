stamp='20231209-222440'
analysis_dir='/home/shared/icon/analysis/icon_weno_'$stamp'_light'
restore_dir='/home/shared/icon/save/user/ckpts/icon_weno/'$stamp
test_data_dirs='/home/shared/icon/data/data0904_weno_cubic_test_light' # use the light version for quick analysis


CUDA_VISIBLE_DEVICES=0 python3 analysis.py --model 'icon_lm' --backend jax --task weno_cubic --write quest,demo \
  --test_caption_id_list -1 --test_data_dirs $test_data_dirs --loss_mode nocap \
  --test_config_filename 'test_lm_weno_config.json' --model_config_filename 'model_lm_config.json' \
  --restore_dir $restore_dir --analysis_dir $analysis_dir --batch_size 64 \
  >out_analysis_icon_weno_$stamp.light.log 2>&1 &&


echo "Done."
