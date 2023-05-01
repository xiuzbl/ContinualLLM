from tools.prompter import Prompter
from tools.get_optimizer import get_optimizer
from tools.get_scheduler import get_scheduler
from utils import  generate_and_tokenize_prompt, DataCollatorForSupervisedDataset, SupervisedDataset, smart_tokenizer_and_embedding_resize
from settings import args
import pdb
from torch.optim import AdamW

from typing import Optional, Dict, Sequence, List
# from accelerate import logging
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate.state import AcceleratorState
import math, json, copy
import os
import sys
from typing import List
from tools.logger import Logger
import fire
import torch
import transformers
from datasets import load_dataset
from mypeft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = get_logger(__name__, log_level="INFO")
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def to_json(config):
    return json.dumps(config, indent=4, sort_keys=True)

def to_dict(args):
    # print(f'args {args}', flush=True)
    config = vars(copy.deepcopy(args))
    for k, v in config.items():
        config[k] = repr(v)
    # print(f'new args {args}', flush=True)
    return config

def calculate_tunable_ratio(model, logger):
    #* Calculate number of parameters 
    num_p = sum([p.numel() for p in model.parameters()])
    tunable_num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_list = []
    for (param_name, param) in model.named_parameters():
        if param.requires_grad:
            trainable_list.append(param_name)
    print(f'Total trainable parameter names {len(trainable_list)}', flush=True)
    logger.info(f'Trainable list>>\n {trainable_list}', main_process_only=True)
    logger.info('Number of parameters: {}'.format(num_p), main_process_only=True)
    logger.info(f'Number of tunable params: {tunable_num_p}, tunable ratio is {"%.6f"%(tunable_num_p/num_p*100)} %', main_process_only=True)
    return

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def save_model(model, tokenizer, save_folder, accelerator):
    # pdb.set_trace()
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_folder, WEIGHTS_NAME)
    output_config_file = os.path.join(save_folder, CONFIG_NAME)

    # torch.save(model_to_save, output_model_file)
    state_dict = model.state_dict()
    # modelpath = output_dir+'/model_last.pt'
    accelerator.save(state_dict, output_model_file)
    # model.save_pretrained(save_folder)
    output_config_file = os.path.join(save_folder, CONFIG_NAME)
    model_to_save.config.to_json_file(output_config_file)

    tokenizer.save_vocabulary(save_folder)

def main(args):
    accelerator.print(f"AcceleratorState:: {AcceleratorState()}")
    trackconfig = {
        'num_iterations':args.num_epochs, 
        'learning_rate': args.lr,
        }
    accelerator.init_trackers(args.exp)
    device = accelerator.device
    # prompt_template_name = 'alpaca'
    # prompter = Prompter(prompt_template_name)

    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map={"": device})

    if args.add_lora:
        logger.info('Add LoRA to the backbone model...', main_process_only=True)
        loraconfig = LoraConfig(
            r=args.lora_r,
            lora_alpha=16,
            target_modules=['q_proj','k_proj','v_proj','o_proj'],
            lora_dropout=0.0,
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, loraconfig)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    tokenizer = LlamaTokenizer.from_pretrained(
                    args.model_name_or_path, 
                    model_max_length=args.model_max_length,
                    padding_size='right',
                    use_fast=False,
                )
    # print(f'pad_token {tokenizer.pad_token}', flush=True)
    if tokenizer.pad_token is None:
        model, tokenizer = smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # print(f'pad_token {tokenizer.pad_token}', flush=True)
    # if "llama" in args.model_name_or_path.lower():
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)

    #* Prepare the data 
    # data_path = args.data_path
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    # if args.val_size > 0:
    #     train_val = data["train"].train_test_split(test_size=args.val_size, shuffle=True, seed=42)
    #     train_data = (
    #         train_val["train"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, prompter, tokenizer, args)))
    #     val_data = (
    #         train_val["test"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, prompter, tokenizer, args)))
    # else:
    #     train_data = data["train"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, prompter, tokenizer, args))
    #     val_data = None
    # data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #         )
    # training_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True, 
    #                         batch_size=args.per_device_train_batch_size,
    #                         collate_fn=lambda x: x,
    #                         num_workers=0)

    logger.info(f"Prepare the dataloaders...", main_process_only=True)

    datadict = make_supervised_data_module(tokenizer, args) # pass in data_path
    train_data = datadict['train_dataset']
    data_collator = datadict['data_collator']

    training_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True, 
                            batch_size=args.per_device_train_batch_size, collate_fn=data_collator)

    num_update_steps_per_epoch = max(math.ceil(len(training_dataloader) // args.gradient_accumulation_steps), 1)
    if args.max_train_steps is None:
        args.max_train_steps = math.ceil(args.num_epochs * num_update_steps_per_epoch)
    else:
        args.num_epochs = args.max_train_steps // num_update_steps_per_epoch + int(
                          args.max_train_steps % num_update_steps_per_epoch > 0
                        )

    #* Set optimizer and scheduler, prepare them with accelerator.
    model, optimizer, trainable_param_names = get_optimizer(model, args)
    # scheduler = get_scheduler(optimizer=optimizer, config=args)
    # scheduler = get_linear_schedule_with_warmup(
    #                 optimizer=optimizer, 
    #                 num_warmup_steps=args.warmup_steps, 
    #                 num_training_steps=args.max_train_steps
                # )
    # optimizer = AdamW(params=model.parameters(), lr=args.lr) 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=args.warmup_ratio*args.max_train_steps, 
        num_training_steps=args.max_train_steps
    )
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    ) 
    model = model.to(device)

    # TODO: Training ---------------------------------------------------------------
    logger.info(f"{'*'*20} Begin Training {'*'*20}", main_process_only=True)
    calculate_tunable_ratio(model, logger)
    logger.info(f"Num training examples = {len(train_data)}", main_process_only=True)
    # logger.info(f"  Num testing examples = {len(val_data)}")
    logger.info(f"Num Epochs = {args.num_epochs}", main_process_only=True)
    logger.info(f"Instantaneous batch size per device = {args.per_device_train_batch_size}", main_process_only=True)
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}", main_process_only=True)
    logger.info(f"Total optimization steps = {args.max_train_steps}")
    logger.info(f"Number optimization steps per epoch = {num_update_steps_per_epoch}", main_process_only=True)
    logger.info(f"Number batches per epoch = {len(training_dataloader)}", main_process_only=True)

    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        logger.info(f"{'*'*10} EPOCH {epoch} {'*'*20}", main_process_only=True)

        for step, batch in enumerate(training_dataloader):
            # if step == 0:
            #     print(f'batch {batch}', flush=True)

            # with accelerator.accumulate(model):
            # pdb.set_trace()
            if True:
                # print(batch, flush=True)

                # outputs = model(**batch)
                # batch = batch.to(device)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                print(f'loss: {type(loss)} {loss}', flush=True)
                accelerator.backward(loss)
                # loss.backward()

                hh = 0
                if step % args.gradient_accumulation_steps == 0:
                    # if accelerator.is_local_main_process:
                    #     for (param_name, param) in model.named_parameters():
                    #         if hh<10:
                    #             if param.requires_grad:
                    #                 print(param_name, param, flush=True)
                    #                 hh += 1

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()

                step_ratio = float(step/len(training_dataloader)*100)
                curr_lr = float(scheduler.get_last_lr()[0])
                
                if step % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    accelerator.log({"epoch":epoch, "learned_data_ratio": step_ratio, "training_loss": loss, "lr":curr_lr}, step=global_step)

                if step % args.print_steps == 0:
                    logger.info(f'EPOCH:{epoch}; GLOBAL STEP: {global_step}; STEP per epoch: {step}; learned_data_ratio: {"%.4f"%step_ratio}; training_loss: {loss}; lr:{"%.8f"%curr_lr}', main_process_only=True)

    #     # TODO: Save model checkpoint after each epoch training.
    #     accelerator.wait_for_everyone()
    #     if True:
    #     # if device == 0:
    #     # if accelerator.is_local_main_process:
    #         model0 = accelerator.unwrap_model(model)
    #         save_folder = os.path.join(args.output_dir,'epoch_'+str(epoch))            
    #         os.makedirs(args.output_dir, exist_ok=True)
    #         os.makedirs(save_folder, exist_ok=True)
    #         # save_model(model, tokenizer, save_folder, accelerator)
    #         model_to_save = model0.module if hasattr(model0, 'module') else model0
    #         CONFIG_NAME = "config.json"
    #         WEIGHTS_NAME = "pytorch_model.bin"
    #         output_model_file = os.path.join(save_folder, WEIGHTS_NAME)
    #         output_config_file = os.path.join(save_folder, CONFIG_NAME)

    #         torch.save(model_to_save, output_model_file)
    #         # state_dict = model.state_dict()
    #         # accelerator.save(state_dict, output_model_file)
    #         # model.save_pretrained(save_folder)
    #         output_config_file = os.path.join(save_folder, CONFIG_NAME)
    #         model_to_save.config.to_json_file(output_config_file)

    #         tokenizer.save_vocabulary(save_folder)

    #         if args.add_lora:
    #             model.save_pretrained(os.path.join(args.output_dir,'epoch_'+str(epoch)))
    #         logger.info(f'Saving model to {save_folder}', main_process_only=True)

    # # TODO: Save the model (and LoRA module) after training.-------------------------------------------------
    # accelerator.wait_for_everyone()
    # logger.info(f'Finish training! Begin saving model to {args.output_dir}', main_process_only=True)
    # # if accelerator.is_local_main_process:
    # if True:
    #     # pdb.set_trace()
    #     model = accelerator.unwrap_model(model)
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     save_folder = args.output_dir
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     CONFIG_NAME = "config.json"
    #     WEIGHTS_NAME = "pytorch_model.bin"
    #     output_model_file = os.path.join(save_folder, WEIGHTS_NAME)
    #     output_config_file = os.path.join(save_folder, CONFIG_NAME)
    #     torch.save(model_to_save, output_model_file)
    #     output_config_file = os.path.join(save_folder, CONFIG_NAME)
    #     model_to_save.config.to_json_file(output_config_file)

    #     tokenizer.save_vocabulary(save_folder)
    #     # save_model(model, tokenizer, args.output_dir, accelerator)

    #     if args.add_lora:
    #         model.save_pretrained(args.output_dir)

    # # print('LAST SAVING FINISH...', flush=True)
    logger.info(f"{'*'*20} FINISHED! CONGRATS! {'*'*20}", main_process_only=True)

    accelerator.end_training()
    
if __name__ == "__main__":
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["wandb"],
        )
    filename = os.path.join(args.output_dir, 'myargs.json')
    with open(filename, "w") as fout:
        arg_dict = to_dict(args)
        fout.write(to_json(arg_dict))
        fout.write('\n')
     
    fire.Fire(main(args))
