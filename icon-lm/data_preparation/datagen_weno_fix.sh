# generate data for fixed operator, used for pretraining classic operator learning

gpu=0

dir=data0604_weno_cubic_fix_0.2_0.2_0.2
traineqns=100
trainnum=100
testeqns=10 # only for visualization during training
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum --dt 0.0005 --file_split 1  --eqn_mode fix_0.2_0.2_0.2  --truncate 100 --seed 101 && 
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name train --eqns $traineqns --num $trainnum --dt 0.0005 --file_split 10 --eqn_mode fix_0.2_0.2_0.2  --truncate 100 --seed 1 &&

echo "Done"