# This sh file focuses on Problem #14 the inverse nonlinear reaction-diffusion PDE problem
# It will restore the trained model from restore_dir, run analysis on the test set of Problem #14, and store results in the analysis directory


stamp='20231005-094726'
analysis_dir='/home/shared/icon/analysis/icon_lm_learn_'$stamp'-pde3-inverse'
restore_dir='/home/shared/icon/save/user/ckpts/icon_lm_learn/'$stamp


CUDA_VISIBLE_DEVICES=0 python3 analysis.py --backend jax --model icon_lm --task ind --write quest,demo,equation \
 --test_demo_num_list 5 --test_caption_id_list -1 --loss_mode nocap \
 --test_config_filename test_lm_pde_full_config.json \
 --test_data_globs 'test_pde_cubic_spatial_inverse*' \
 --model_config_filename model_lm_config.json \
 --analysis_dir $analysis_dir \
 --restore_dir $restore_dir \
 --batch_size 10 >out_analysis_icon_lm_learn-$stamp-pde3-inverse.log 2>&1 &&


echo "Done."
