gpu=0

dir=data0904_weno_cubic
traineqns=1000
trainnum=100
testeqns=10 # only for visualization during training
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum --dt 0.0005 --file_split 1  --truncate 100 --seed 101 && 
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name train --eqns $traineqns --num $trainnum --dt 0.0005 --file_split 10 --truncate 100 --seed 1 &&

# for analysis
dir=data0904_weno_cubic_test
testeqns=11
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --seed 101 &&

# for quick analysis
dir=data0904_weno_cubic_test_light
testeqns=5
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --seed 101 &&

echo "Done"

