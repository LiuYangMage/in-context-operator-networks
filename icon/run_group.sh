gpu=0

k_dim=3
k_mode=itx
problem=group-$k_dim-$k_mode

CUDA_VISIBLE_DEVICES=$gpu python3 run.py --problem $problem --num_heads 8 --num_layers 6 --hidden_dim 256 --train_batch_size 16 --epochs 20 --train_warmup_percent 40  --train_data_dirs './data_generation/data0511a' --k_dim $k_dim --k_mode $k_mode --train_data_globs 'train*ode*linear1*' --test_data_globs 'test*ode*linear1*' --tfboard >out_0511a_${problem}_odes_l1.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 run.py --problem $problem --num_heads 8 --num_layers 6 --hidden_dim 256 --train_batch_size 16 --epochs 20 --train_warmup_percent 40  --train_data_dirs './data_generation/data0511a' --k_dim $k_dim --k_mode $k_mode --train_data_globs 'train*ode*linear1*','train*ode*const*' --test_data_globs 'test*ode*linear1*','test*ode*const*' --tfboard >out_0511a_${problem}_odes_l1c.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 run.py --problem $problem --num_heads 8 --num_layers 6 --hidden_dim 256 --train_batch_size 16 --epochs 20 --train_warmup_percent 40  --train_data_dirs './data_generation/data0511a' --k_dim $k_dim --k_mode $k_mode --train_data_globs 'train*ode*linear1*','train*ode*linear2*' --test_data_globs 'test*ode*linear1*','test*ode*linear2*' --tfboard >out_0511a_${problem}_odes_l1l2.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 run.py --problem $problem --num_heads 8 --num_layers 6 --hidden_dim 256 --train_batch_size 16 --epochs 20 --train_warmup_percent 40  --train_data_dirs './data_generation/data0511a' --k_dim $k_dim --k_mode $k_mode --train_data_globs 'train*ode*' --test_data_globs 'test*ode*' --tfboard >out_0511a_${problem}_odes_l1l2c.log 2>&1 &&

echo "Done"