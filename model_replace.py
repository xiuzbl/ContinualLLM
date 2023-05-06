import os
import sys
from utils import generate_and_tokenize_prompt, DataCollatorForSupervisedDataset, SupervisedDataset, smart_tokenizer_and_embedding_resize, preprocess
from tools.prompter import Prompter
# import fire
# import gradio as gr
import torch
import transformers
from mypeft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import argparse

def save(model, tokenizer, save_folder):
    # dist.barrier()
    model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save = model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_folder, WEIGHTS_NAME)
    output_config_file = os.path.join(save_folder, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    print(f'Finished saving the model to {output_model_file}', flush=True)
    output_config_file = os.path.join(save_folder, CONFIG_NAME)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_folder)
    print(f'Finished saving the tokenizer.', flush=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument('--pretrained_model_path', type=str)
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--tokenizer_path', type=str)
parser.add_argument('--output_dir', type=str)

parser.add_argument('--add_lora', type=bool, default=False)
parser.add_argument('--only_pretrained_model', type=bool, default=False)
parser.add_argument('--lora_weights', type=str, default='tloen/alpaca-lora-7b')
parser.add_argument('--replace_lora_weights', type=bool, default=False)
args = parser.parse_args()

prompt_template = ""  # The prompt template to use, will default to alpaca.
prompter = Prompter(prompt_template)
server_name = "0.0.0.0"  # Allows to listen on all interfaces by providing '0.
share_gradio = False

tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

# if args.add_lora:
if True:
    model = LlamaForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map={"":device}
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    model, tokenizer = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )    

    print('Here! Load from pretrained lora weights', flush=True)
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        device_map={"": device},
        replace_lora_weights=True,
    )

    save_folder = os.path.join(args.output_dir, 'model_replaced')            
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    save(model, tokenizer, save_folder)

print(f'Congrats!!!')