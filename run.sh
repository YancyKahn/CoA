conda activate llm-attacks

python=/home/ecs-user/.conda/envs/llm-attacks/bin/python
mianfile=/home/ecs-user/project-yxk/llms-attacks/multi-round-attacks/experiment.py
apiserverfile=/home/ecs-user/project-yxk/llms-attacks/multi-round-attacks/fastapi/fast_api.py
logdir=/home/ecs-user/project-yxk/llms-attacks/logs/output/

% start apiserver
$python $apiserverfile --model-name "lmsys/vicuna-13b-v1.5-16k" > $logdir/apiserver.log 2>&1 &
$python $mianfile --attack-modle "vicuna-api" --target-model "vicuna-api" --judge-model "vicuna-api" > $logdir/vicuna.log 2>&1 &
% end apiserver
kill -9 $(ps -ef | grep fast_api.py | grep -v grep | awk '{print $2}')

% start apiserver
$python $apiserverfile --model-name "lmsys/vicuna-13b-v1.5-16k" "meta-llama/Llama-2-7b-chat-hf" > $logdir/apiserver.log 2>&1 &
$python $mianfile --attack-modle "vicuna-api" --target-model "llama2-api" --judge-model "vicuna-api" > $logdir/llama2.log 2>&1 &
% end apiserver
kill -9 $(ps -ef | grep fast_api.py | grep -v grep | awk '{print $2}')

% start apiserver
%$python $apiserverfile --model-name "lmsys/vicuna-13b-v1.5-16k" "THUDM/chatglm2-6b" > $logdir/apiserver.log 2>&1 &
%$python $mianfile --attack-modle "vicuna-api" --target-model "chatglm-api" --judge-model "vicuna-api" > $logdir/chatglm.log 2>&1 &
% end apiserver
%kill -9 $(ps -ef | grep fast_api.py | grep -v grep | awk '{print $2}')

% start apiserver
%$python $apiserverfile --model-name "lmsys/vicuna-13b-v1.5-16k" "HuggingFaceH4/zephyr-7b-beta" > $logdir/apiserver.log 2>&1 &
%$python $mianfile --attack-modle "vicuna-api" --target-model "zephyr-api" --judge-model "vicuna-api" > $logdir/zephyr-chatglm.log 2>&1 &
% end apiserver
%kill -9 $(ps -ef | grep fast_api.py | grep -v grep | awk '{print $2}')