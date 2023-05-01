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
# from utils.callbacks import Iteratorize, Stream

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
parser.add_argument('--add_lora', type=bool, default=False)
parser.add_argument('--only_pretrained_model', type=bool, default=False)
parser.add_argument('--lora_weights', type=str, default='tloen/alpaca-lora-7b')
parser.add_argument('--replace_lora_weights', type=bool, default=False)
args = parser.parse_args()

prompt_template = ""  # The prompt template to use, will default to alpaca.
prompter = Prompter(prompt_template)
server_name = "0.0.0.0"  # Allows to listen on all interfaces by providing '0.
share_gradio = False

print(f'Begin loading tokenizer~', flush=True)
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
print(f'Finish loading tokenizer~', flush=True)

print(f'Begin loading the model~', flush=True)
if args.add_lora:
    model = LlamaForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map={"":device}
    )
    model, tokenizer = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
    print('Here! Load from pretrained lora weights', flush=True)
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        device_map={"": device},
        replace_lora_weights=args.replace_lora_weights,
    )
elif args.only_pretrained_model:
    model = LlamaForCausalLM.from_pretrained(
         args.pretrained_model_path,
        device_map={"":device}
    )
else:
    # model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native")
    # model = LlamaForCausalLM.from_pretrained(
    #      args.pretrained_model_path,
    #     device_map={"":device}
    # )
    # model, tokenizer = smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    # model.load_state_dict(args.model_name_or_path)
    print(f'Begin loading the trained checkpoint~', flush=True)
    # ckp_state_dict = torch.load(args.model_name_or_path, map_location=device)
    # model.load_state_dict(ckp_state_dict, strict=True)
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map={"":device}
    )
    print(f'Finish loading the trained checkpoint~', flush=True)
    # model = model.cuda()
# model.print_trainable_parameters()

print(f'Finish loading the model~', flush=True)

# if not args.only_pretrained_model:
# # if True:
#     print(f'pad_token {tokenizer.pad_token}', flush=True)
#     tokenizer.add_special_tokens(
#         {
#             "eos_token": DEFAULT_EOS_TOKEN,
#             "bos_token": DEFAULT_BOS_TOKEN,
#             "unk_token": DEFAULT_UNK_TOKEN,
#         }
#     )

 
model.eval()

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    # temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    # max_new_tokens=500,
    stream_output=False,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    print(f'Input sentence: {prompt}', flush=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # print(f'output: {output}', flush=True)

    # yield prompter.get_response(output)
    return prompter.get_response(output)


# TODO Evaluating----------------------------------------------------------------
while True:
# while False:
    # instruction = "List which tasks you can solve as a language model."
    # print("Response:\n", evaluate(instruction, stream_output=False))
    print(f'{"-"*20}', flush=True)
    print(f'Will you need the instruction? Yes or No', flush=True)
    give_instruction = str(input())
    if give_instruction in ['Yes','yes','ture','YES','Y','y','T','t']:
        print(f'Give me your instruction>>', flush=True)
        instruction = str(input())
    else:
        instruction = "Translate the following sentence from English to Chinese."
        print(f'Use the instruction {instruction}',flush=True)

    print(f'{"-"*20}', flush=True)
    print(f'Will you need the input? Yes or No', flush=True)
    give_input = str(input())
    if give_input in ['Yes','yes','ture','YES','Y','y','T','t']:
        print(f'Give me your input>>', flush=True)
        input_sentence = str(input())
    else:
        input_sentence = None

    print("Response>>\n", evaluate(instruction, input=input_sentence, stream_output=False))
    print(f'{"*"*20}', flush=True)
# instruction = "List which tasks you can solve as a language model."
# instruction = "Give some diverse examples for question-answering tasks."
# instruction = "Evaluate the above examples, are they correctly answered?"

# for instruction in [
#     "Tell me about alpacas.",
#     "Tell me about the president of Mexico in 2019.",
#     "Tell me about the king of France in 2019.",
#     "List all Canadian provinces in alphabetical order.",
#     "Write a Python program that prints the first 10 Fibonacci numbers.",
#     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'."
#     "Tell me five words that rhyme with 'shock'.",
#     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
#     "Count up from 1 to 500.",
# ]:
