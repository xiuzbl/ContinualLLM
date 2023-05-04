port=60000
now="$(date +'%m%d')"
echo DATE: $now
# dir=root
# dir=mnt
eflopdir=/mnt/user/E-zhaoyingxiu.zyx-354256
# exp=${now}_test_lr3e-4
job=debug
# job=lora_debug
# exp=$job
# job=llama_gpt4_run2
# job=llama_gpt4_lr1e-5_run1
# job=llama_gpt4_1epoch
# exp=${now}${dir}_$job
exp=${now}eflop_$job
echo Experiment: $exp

# gpuid=0,1
# ngpu=2 #! debug
gpuid=0,1,2,3,4,5,6,7
ngpu=8 #! dlc
# total batch size 256=ngpu*trainbtz*grad_accu=8*4*8
# total batch size 128=ngpu*trainbtz*grad_accu=8*2*8
# 128
trainbtz=2
# trainbtz=8
evalbtz=4
grad_accu=4
# grad_accu=8
# grad_accu=1
# grad_accu=2
# add_lora=true #!
add_lora=
echo train_batch_size: $trainbtz
echo gradient_accumulation_step: $grad_accu
echo Add LoRA: $add_lora

# epoch=10
# epoch=2
epoch=5
# epoch=3
lr=2e-5 # tune llama
# lr=1e-4
# lr=1e-4 # tune lora
warmup_ratio=0.03
# warmup_ratio=0.0
# warmup_steps=100
warmup_steps=0
# warmup_ratio=0.1
echo epoch: $epoch
echo lr: $lr
echo warmup_ratio: $warmup_ratio
echo warmup_steps: $warmup_steps

# max_length=2048
max_length=1024
print_steps=20
# print_steps=1

# datapath=$eflopdir/LLMDATA/alpaca_data_gpt4.json
# datapath=$eflopdir/LLMDATA/mixed_inst_trans.json
datapath=$eflopdir/LLMDATA/debug.json
# datapath=$eflopdir/LLMDATA/wmt19_en-zh.json
# datapath=$eflopdir/LLMDATA/wmt_test.json
modelpath=$eflopdir/MODELS/llama-7b-hf
# modelpath=$eflopdir/MODELS/chavinlo-alpaca-native # 52K self-instructions SFT trained LLaMA

echo DATA: $datapath

output=$eflopdir/LLMOUT/outputs/$exp
log=$eflopdir/LLMOUT/logs/$exp.log
tbdir=$eflopdir/LLMOUT/tblogs/$exp
rm -rf $tbdir
mkdir -p $eflopdir/LLMOUT/outputs $output $eflopdir/LLMOUT/logs $eflopdir/LLMOUT/tblogs $tbdir

file=$eflopdir/CODE/ContinualLLM/dsrun.py
# $eflopdir/xiuenvs/alpaca3.9/bin/wandb login b98a11d4688e1fb9308f6aaaed3e119d11a91aa4
# run=$eflopdir/xiuenvs/alpaca3.9/bin/accelerate
env=/mnt/user/E-zhaoyingxiu.zyx-354256/envs/py39/bin
$env/wandb login b98a11d4688e1fb9308f6aaaed3e119d11a91aa4
# run=accelerate
# run=deepspeed
run=$env/deepspeed
ds_config=configs/ds_config.json
echo deepspeed config $ds_config

# $run config --config_file $eflopdir/accelerate_config.yaml
    # --mixed_precision=bf16 --num_processes=$ngpu --num_machines=1 \

echo Begin training...
export CUDA_VISIBLE_DEVICES=$gpuid
$run \
    --num_gpus=$ngpu \
    $file \
    --deepspeed \
    --deepspeed_config=$ds_config \
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
    --model_max_length=$max_length \
    --print_steps=$print_steps \
    >$log 2>&1
