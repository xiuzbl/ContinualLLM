port=60000
now="$(date +'%m%d')"
echo DATE: $now
# dir=root
dir=mnt
# exp=${now}_test_lr3e-4
job=debug
exp=$job
# job=llama_gpt4_run2
# job=llama_gpt4_run0a
# exp=${now}${dir}_$job
echo Experiment: $exp

ngpu=2 #! debug
# ngpu=8 #! dlc
# total batch size 128=ngpu*trainbtz*grad_accu=8*2*8
# 256
trainbtz=4
# trainbtz=8
evalbtz=4
# grad_accu=4
# grad_accu=8
grad_accu=2
# add_lora=true #!
add_lora=
echo Add LoRA: $add_lora

# epoch=10
epoch=3
# poch=5
# epoch=2
lr=1e-5
# lr=3e-4
# warmup_ratio=0.03
warmup_ratio=0.0
# warmup_steps=100
warmup_steps=0
# warmup_ratio=0.1
echo epoch: $epoch
echo lr: $lr
echo warmup_ratio: $warmup_ratio
echo warmup_steps: $warmup_steps

# datapath=/$dir/data/yingxiu/LLMDATA/alpaca_data_gpt4.json
datapath=/$dir/data/yingxiu/LLMDATA/debug.json
# datapath=/$dir/data/yingxiu/LLMDATA/wmt19_en-zh.json
# datapath=/$dir/data/yingxiu/LLMDATA/wmt_test.json
modelpath=/$dir/data/yingxiu/llama-7b-hf

echo DATA: $datapath

output=/$dir/data/yingxiu/LLMOUT/outputs/$exp
log=/$dir/data/yingxiu/LLMOUT/logs/$exp.log
tbdir=/$dir/data/yingxiu/LLMOUT/tblogs/$exp
mkdir -p /$dir/data/yingxiu/LLMOUT/outputs $output /$dir/data/yingxiu/LLMOUT/logs /$dir/data/yingxiu/LLMOUT/tblogs $tbdir

file=/$dir/data/yingxiu/LLMCODE/run.py
# /$dir/data/yingxiu/xiuenvs/alpaca3.9/bin/wandb login b98a11d4688e1fb9308f6aaaed3e119d11a91aa4
# run=/$dir/data/yingxiu/xiuenvs/alpaca3.9/bin/accelerate
wandb login b98a11d4688e1fb9308f6aaaed3e119d11a91aa4
run=accelerate

# $run config --config_file /$dir/data/yingxiu/accelerate_config.yaml
    # --mixed_precision=bf16 --num_processes=$ngpu --num_machines=1 \

echo Begin training...
$run launch \
    --num_processes=$ngpu --num_machines=1 \
    --config_file configs/accelerate_config_${ngpu}gpu.yaml \
    $file \
    --output_dir=$output \
    --exp=$exp \
    --data_path=$datapath \
    --model_name_or_path=$modelpath \
    --per_device_train_batch_size=$trainbtz \
    --per_device_eval_batch_size=$evalbtz \
    --gradient_accumulation_steps=$grad_accu \
    --lr=$lr \
    --warmup_ratio=$warmup_ratio \
    --warmup_steps=$warmup_steps \
    --num_epochs=$epoch \
    --logging_dir=$tbdir \
    --add_lora=$add_lora \
    >$log 2>&1



