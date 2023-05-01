# dir=root
gpuid=1
echo GPUID $gpuid
dir=mnt
# add_lora=true
# replace_lora=true
add_lora=
replace_lora=

log=/$dir/data/yingxiu/alpaca-lora/logs/eval_test.log
py=/$dir/data/yingxiu/xiuenvs/alpaca3.9/bin/python

# prt_model=/$dir/data/yingxiu/huggyllama-7b
prt_model=/$dir/data/yingxiu/llama-7b-hf
# prt_model=/$dir/data/yingxiu/chavinlo-alpaca-native
# only_prt=true
# model_path=$prt_model

only_prt=

# model_path=/$dir/data/yingxiu/llama-7b-hf
# model_path=chavinlo/alpaca-native
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/debug
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0423root_llama_gpt4_run0/epoch_1
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0423root_llama_gpt4_run0/
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0424root_llama_cleaned_run0/epoch_2
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0424root_llama_cleaned_lr1e-5_run0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0418root_llama_gpt4_run3a
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0418root_llama_gpt4_run4/epoch_0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0419root_llama_gpt4_run4a/epoch_0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0419mnt_llama_gpt4_run0/epoch_0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0417root_llama_gpt4_run1/epoch_1
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0418root_llama_gpt4_run2/epoch_2
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0421root_llama_gpt4_run0/epoch_0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0421root_llama_gpt4_run0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/lora_debug/epoch_0
# lorapath=/mnt/data/yingxiu/LLMOUT/outputs/lora_debug/epoch_0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0424root_lora_mix_run0/epoch_2
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0424root_llama_gpt4_run1
model_path=/mnt/data/yingxiu/LLMOUT/outputs/0424root_llama_cleaned_lr1e-5_run0
# model_path=/mnt/data/yingxiu/LLMOUT/outputs/0424root_lora_mix_run1/epoch_2
lorapath=$model_path

# lorapath=/$dir/data/yingxiu/alpaca-lora-7b/adapter_model.bin
# lorapath=/$dir/data/yingxiu/tloen_alpaca-lora-7b/adapter_model.bin
# lorapath=/mnt/data/yingxiu/alpaca-lora/outputs/0414_alpaca_mylora_1epoch
# lorapath=/mnt/data/yingxiu/alpaca-lora/outputs/0414_alpaca_mylora_1epoch/checkpoint-400
# lorapath=/mnt/data/yingxiu/alpaca-lora/outputs/0414_alpaca_mylora_1epoch/checkpoint-400/adapter_model.bin
# lorapath=/mnt/data/yingxiu/alpaca-lora/outputs/0414_alpaca_mylora_1epoch/adapter_model.bin
# lorapath=/mnt/data/yingxiu/LLMOUT/outputs/0415_debug/epoch_4
# lorapath=/mnt/data/yingxiu/LLMOUT/outputs/0415_wmttest01/epoch_1

# tokenizer_path=/mnt/data/yingxiu/LLMOUT/outputs/0415_wmttest01
# tokenizer_path=/mnt/data/yingxiu/LLMOUT/outputs/0418root_llama_gpt4_run2/
# tokenizer_path=chavinlo/alpaca-native
# tokenizer_path=/mnt/data/yingxiu/LLMOUT/outputs/debug
tokenizer_path=$model_path
# echo epoch 1

echo add_lora: $add_lora
echo only pretrained model: $only_prt
echo pretrained_model: $prt_model
echo model_path: $model_path
echo tokenizer_path: $tokenizer_path

echo Begin Generating...
# py=python
file=/$dir/data/yingxiu/LLMCODE/generate.py
CUDA_VISIBLE_DEVICES=$gpuid \
$py $file \
    --model_name_or_path=${model_path}/pytorch_model.bin \
    --pretrained_model_path=$prt_model \
    --add_lora=$add_lora \
    --lora_weights=$lorapath \
    --replace_lora_weights=$replace_lora \
    --only_pretrained_model=$only_prt \
    --tokenizer_path=$tokenizer_path 
    # >$log 2>&1

echo Finish! Congrats!