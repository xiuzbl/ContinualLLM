import argparse
from transformers import SchedulerType
import deepspeed

parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument('--add_lora', type=bool, default=False)
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--exp', type=str, default='test')
parser.add_argument('--max_train_steps', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--per_device_train_batch_size', type=int, default=4)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--print_steps', type=int, default=20)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--warmup_ratio', type=float, default=0.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--scheduler', type=str, default="linear_decay_with_warmup")
parser.add_argument('--val_size', type=int, default=2000)
parser.add_argument('--logging_dir', type=str, default=None)
parser.add_argument('--model_max_length', type=int, default=512)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()