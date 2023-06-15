gpu=0
prefix=data0520

# light dataset
quests=1
ood_coeff1_grids=10
ood_coeff2_grids=11
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --eqn_types ode_auto_const --dir ./${prefix}_ood_odeconst_light >out_data_ood_ode_auto_const_light.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --eqn_types ode_auto_linear1 --dir ./${prefix}_ood_odelinear1_light >out_data_ood_ode_auto_linear1_light.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --eqn_types ode_auto_linear2 --dir ./${prefix}_ood_odelinear2_light >out_data_ood_ode_auto_linear2_light.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --length 100 --dx 0.01 --eqn_types pde_porous_spatial --dir ./${prefix}_ood_pdeporous_randbdry_light >out_data_ood_pde_porous_spatial_light.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --length 100 --dt 0.01 --eqn_types series_damped_oscillator --dir ./${prefix}_ood_seriesdamped_light >out_data_ood_seriesdamped_light.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --eqn_types ode_auto_linear3 --dir ./${prefix}_nt_odelinear3_light >out_data_nt_ode_auto_linear3_light.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --ood_coeff1_grids $ood_coeff1_grids --ood_coeff2_grids $ood_coeff2_grids --quests $quests --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types ot_rho1param --dir ./${prefix}_nt_ot_rho1param_light >out_data_nt_ot_rho1param_light.log 2>&1 &&

# formal dataset
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --eqn_types ode_auto_const --dir ./${prefix}_ood_odeconst >out_data_ood_ode_auto_const.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --eqn_types ode_auto_linear1 --dir ./${prefix}_ood_odelinear1 >out_data_ood_ode_auto_linear1.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --eqn_types ode_auto_linear2 --dir ./${prefix}_ood_odelinear2 >out_data_ood_ode_auto_linear2.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --length 100 --dx 0.01 --eqn_types pde_porous_spatial --dir ./${prefix}_ood_pdeporous_randbdry >out_data_ood_pde_porous_spatial.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --length 100 --dt 0.01 --eqn_types series_damped_oscillator --dir ./${prefix}_ood_seriesdamped >out_data_ood_seriesdamped.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --eqn_types ode_auto_linear3 --ood_coeff2_grids 201 --dir ./${prefix}_nt_odelinear3 >out_data_nt_ode_auto_linear3.log 2>&1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_ood.py --seed 202 --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types ot_rho1param --dir ./${prefix}_nt_ot_rho1param >out_data_nt_ot_rho1param.log 2>&1 &&


echo "Done"



