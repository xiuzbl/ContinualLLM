from tools.prompter import Prompter
from tools.get_optimizer import get_optimizer
from tools.get_scheduler import get_scheduler
from utils import  generate_and_tokenize_prompt, DataCollatorForSupervisedDataset, SupervisedDataset, smart_tokenizer_and_embedding_resize
from dsarguments import args
from mypeft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.tensorboard import SummaryWriter
import pdb
import numpy as np
from torch.optim import AdamW
import datetime, wandb, re
from typing import Optional, Dict, Sequence, List
import math, json, copy, random
import os
import sys
from typing import List
from tools.logger import Logger
import fire
import torch
import transformers
from datasets import load_dataset

from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    get_linear_schedule_with_warmup, 
    # get_cosine_schedule_with_warmup,
    set_seed,
)
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import deepspeed
from deepspeed.runtime.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
torch.autograd.set_detect_anomaly(True)
# torch.backends.cuda.matmul.allow_tf32 = True
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float):
    if current_step < num_warmup_steps:
        return max(0.000001, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) 

def to_device(batch, device_id):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device_id)
        except:
            output[k] = v
    return output

# logger = get_logger(__name__, log_level="INFO")
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
    logger.info(f'Trainable list>>\n {trainable_list}')
    logger.info('Number of parameters: {}'.format(num_p))
    logger.info(f'Number of tunable params: {tunable_num_p}, tunable ratio is {"%.6f"%(tunable_num_p/num_p*100)} %')
    return

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

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

def main(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()

    # if rank<=0:
    try:
        wandb.init(project=args.exp, dir=args.logging_dir)
        use_wandb=True
        print(f'Use Wandb to log~', flush=True)
    except:
        use_wandb=False
        print(f'NOT Use Wandb.', flush=True)     


    #* Align different configs.
    deepspeed_config = json.load(open(args.deepspeed_config, 'r', encoding='utf-8'))
    # model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    # assert deepspeed_config['gradient_accumulation_steps'] == args.gradient_accumulation_steps
    assert deepspeed_config['train_micro_batch_size_per_gpu'] == args.per_device_train_batch_size

    # model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map={"": device_id})
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)

    if args.add_lora:
        if rank<=0: logger.info('Add LoRA to the backbone model...')
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
    # print(f'pad_token {tokenizer.pad_token}', flush=True)

    if rank<=0: 
        train_writer = SummaryWriter(os.path.join(args.logging_dir, 'train'), flush_secs=10)
        logger.info(f"Prepare the dataloaders...")

    datadict = make_supervised_data_module(tokenizer, args) # pass in data_path
    train_dataset = datadict['train_dataset']
    data_collator = datadict['data_collator']

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        training_dataloader = DataLoader(train_dataset, collate_fn=data_collator,batch_size=args.per_device_train_batch_size, sampler=train_sampler)
    else:
        training_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=args.per_device_train_batch_size)

    num_update_steps_per_epoch = max(math.ceil(len(training_dataloader) // args.gradient_accumulation_steps), 1)
    if args.max_train_steps > 0:
        args.max_train_steps = math.ceil(args.num_epochs * num_update_steps_per_epoch)
        args.num_epochs = args.max_train_steps // num_update_steps_per_epoch + int(
                          args.max_train_steps % num_update_steps_per_epoch > 0
                        )
    else:
        args.max_train_steps = math.ceil(args.num_epochs * num_update_steps_per_epoch)

    #* Set optimizer and scheduler, prepare them with accelerator.
    # _, optimizer, trainable_param_names = get_optimizer(model, args)
    # scheduler = get_scheduler(optimizer=optimizer, config=args)
    # scheduler = get_linear_schedule_with_warmup(
    #                 optimizer=optimizer, 
    #                 num_warmup_steps=args.warmup_steps, 
    #                 num_training_steps=args.max_train_steps
                # )
    optimizer = AdamW(params=model.parameters(), lr=args.lr) 

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer, 
    #     num_warmup_steps=math.ceil(args.warmup_ratio * args.max_train_steps), 
    #     num_training_steps=args.max_train_steps
    # )
    # print(f'scheduler first {scheduler}', flush=True)
    # model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # model = model.to(device_id)
    deepspeed_config["scheduler"]["params"]["total_num_steps"] = args.max_train_steps
    deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = math.ceil(args.warmup_ratio * args.max_train_steps) 
    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        dist_init_required=False,
        config=deepspeed_config
    )
        # lr_scheduler=scheduler,
        # args=args
    print(f'scheduler {scheduler}', flush=True)
    model.train()
    # wandb.watch(model, log_freq=4)

    # TODO: Training ---------------------------------------------------------------
    if rank <= 0:
        logger.info(f"{'*'*20} Begin Training {'*'*20}")
        calculate_tunable_ratio(model, logger)
        logger.info(f"Num training examples = {len(train_dataset)}")
        # logger.info(f"Num testing examples = {len(val_data)}")
        logger.info(f"Num Epochs = {args.num_epochs}")
        logger.info(f"Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps = {args.max_train_steps}")
        logger.info(f"Number optimization steps per epoch = {num_update_steps_per_epoch}")
        logger.info(f"Number batches per epoch = {len(training_dataloader)}")

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        logger.info(f"{'*'*10} EPOCH {epoch} {'*'*20}")

        steps_in_epoch = len(training_dataloader)
        losses = []
        for step, batch in enumerate(training_dataloader):
            # if step == 0:
            #     print(f'batch {batch}', flush=True)

            # pdb.set_trace()
            print(f'device {model.device}', flush=True)
            batch = to_device(batch, model.device)
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # outputs = model(**batch)
            loss0 = outputs.loss
            # loss = loss0 / args.gradient_accumulation_steps
            model.backward(loss0)
            # model.optimizer.backward(loss)
            # model.micro_steps += 1

            # print(f'batch:{step}; loss: {type(loss)} {loss}', flush=True)

            # if rank <= 0:
            #     logger.info(f'After optimization step memory INFO:') 
            #     logger.info(f'Allocated: {torch.cuda.memory_allocated()}')
            model.step()
            losses.append(loss0.item())
            # if rank<=0: print(f'ds_global_step:{model.global_steps}; ds_lr: {model.lr_scheduler.state_dict()}', flush=True)
            # loss = loss0 / args.gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0 or (steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch):# last step in epoch but step is always smaller than gradient_accumulation_steps
                # model.step()
                # clip_grad_norm_(parameters=model.module.parameters(), max_norm=model.gradient_clipping(), mpu=model.mpu)
                # model.optimizer.step()
                # model.lr_scheduler.step()
                # model.optimizer.zero_grad()
                # model.global_steps += 1
                # model.global_samples += model.train_batch_size() * args.gradient_accumulation_steps

                global_step += 1

                curr_lr = float(scheduler.get_last_lr()[0])
                step_ratio = float(step/len(training_dataloader)*100)
            # if step % args.print_steps == 0:
                if use_wandb: 
                    wandb.log({"epoch":epoch, "learned_data_ratio": step_ratio, "training_loss": float(np.mean(losses)), "lr":curr_lr}, step=global_step)
                if rank <= 0:
                    train_writer.add_scalar('train/loss', float(np.mean(losses)), global_step)
                    train_writer.add_scalar('lr',curr_lr, global_step)
                    logger.info(f'EPOCH:{epoch}; GLOBAL STEP: {global_step}; STEP per epoch: {step}; learned_data_ratio: {"%.4f"%step_ratio}; training_loss_per_batch: {float(np.mean(losses))}; lr:{"%.8f"%curr_lr}')
                losses = []

            if global_step >= args.max_train_steps:
                break

        # TODO: Save model checkpoint after each epoch training.
        if dist.is_initialized():
            dist.barrier()
        if args.output_dir is not None and rank <= 0:
            save_folder = os.path.join(args.output_dir,'epoch_'+str(epoch))            
            logger.info(f'Begin saving model checkpoint to {save_folder}')
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(save_folder, exist_ok=True)
            # logger.info(f'Begin saving model to {save_folder}')
            # save(model, tokenizer, save_folder)
            # model.save_checkpoint(save_folder)
            if args.add_lora:
                model.save_pretrained(os.path.join(args.output_dir,'epoch_'+str(epoch)))
            logger.info(f'Finish saving model checkpoint of epoch {epoch}.')

    # # TODO: Save model checkpoints ---------------------------------
    # if args.output_dir is not None and not args.debug:
    if dist.is_initialized():
        dist.barrier()
    if args.output_dir is not None and rank <= 0:
        logger.info(f'Finish Training the model.') 
        save_folder = os.path.join(args.output_dir)            
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f'Begin saving model to {save_folder}')
        save(model, tokenizer, save_folder)
        if args.add_lora:
            model.save_pretrained(save_folder)

    if rank<=0: logger.info(f"{'*'*20} FINISHED! CONGRATS! {'*'*20}")

if __name__ == "__main__":

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # deepspeed.init_distributed()
        deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(seconds=1800000))
    args.local_rank = int(os.environ['LOCAL_RANK'])
    logger = Logger()

    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.distributed.barrier()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'Finish arguments construction!')

    filename = os.path.join(args.output_dir, 'myargs.json')
    with open(filename, "w") as fout:
        arg_dict = to_dict(args)
        fout.write(to_json(arg_dict))
        fout.write('\n')

    # fire.Fire(main(args))
    main(args)
