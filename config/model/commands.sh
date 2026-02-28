python run_base.py --model_config config/model/eval_dropout_lafan1.yaml --log_file output/base/dropout_lafan1/log.txt --int_output_dir output/base/eval_dropout_lafan1_crps/ --out_
model_file output/base/eval_dropout_lafan1_crps/model_param.pth --mode eval --master_port 29500 --rand_seed 122 --model_path /home/kvats/285/output/base/dropout_lafan1_crps/model_param.pth

python run_base.py --model_config config/model/dropout_lafan1-crps.yaml --log_file output/base/dropout_lafan1/log.txt --int_output_dir output/base/dropout_lafan1_crps/ --out_model_file output/base/dropout_lafan1_crps/model_param.pth --mode train --master_port 29500 --rand_seed 122 --model_path /home/kvats/285/output/base/dropout_lafan1/model_param.pth

python run_base.py --model_config config/model/eval_dropout_lafan1.yaml --log_file output/base/dropout_lafan1/log.txt --int_output_dir output/base/eval_dropout_lafan1/ --out_model
_file output/base/eval_dropout_lafan1_crps/model_param.pth --mode eval --master_port 29500 --rand_seed 122 --model_path /home/kvats/285/output/base/dropout_lafan1/model_param.pth

python run_base.py --model_config config/model/eval_amdm_lafan1.yaml --log_file output/base/dropout_lafan1/log.txt --int_output_dir output/base/eval_amdm_lafan1/ --out_model_file 
output/base/eval_amdm_lafan1/model_param.pth --mode eval --master_port 29500 --rand_seed 122 --model_path /home/kvats/285/output/base/amdm_lafan1/model_param.pth