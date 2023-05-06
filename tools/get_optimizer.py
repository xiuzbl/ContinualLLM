import torch.optim as optim
from transformers import Adafactor
import re
from collections import defaultdict

def get_optimizer(model, config, other_param_names=None):
    # params = list(model.parameters())
    trainable_param_names = set()
    optim_name = config.optimizer
    # param_groups = defaultdict(lambda: {"params": []})
    param_groups = {"params": []}

    for (param_name, param) in model.named_parameters():
        if config.add_lora:
            if 'lm_head' in param_name or 'embed_tokens' in param_name:
                param.requires_grad = False    
        
        if param.requires_grad:
            trainable_param_names.add(param_name)
            param_groups['params'].append(param)

    # param_groups = param_groups.values()
    param_groups = [param_groups]
    if optim_name.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=config.lr)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    elif optim_name.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, eps=1e-8)
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            scale_parameter=config.scale_parameter,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return model, optimizer, trainable_param_names
    

def ori_get_optimizer(model, config, other_param_names=None):
    """
    Construct optimizer based on config

    :param model:
    :param config:
    :return:
    """
    optim_name = config.optimizer
    config.trainable_param_names = ".*"

    def param_name_to_group_name(param_name):
        if False:
        # if True:
            return ".".join(param_name.split(".")[:3])
            # only needed when the model has many trainable parameters, disabled in our expeirments
        else:
            return "."

    param_groups = defaultdict(lambda: {"params": []})
    trainable_param_names = set()
    added_param = []
    for (param_name, param) in model.named_parameters():
        if type(config.trainable_param_names)==list: #* reset trainable parameters rather than that written in the config file.
            # print(11)
            for train_param in config.trainable_param_names:
                if re.fullmatch(train_param, param_name):
                    # print(f'train:{param_name}', flush=True)
                    param.requires_grad = True
                    added_param.append(param_name)
                    param_groups[param_name_to_group_name(param_name)]["params"].append(param)
                    trainable_param_names.add(param_name)
                    # print(f'{param_name} {param.requires_grad}',flush=True)
                elif param_name in added_param:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # print(22)
            if re.fullmatch(config.trainable_param_names, param_name):
                param.requires_grad = True
                added_param.append(param_name)
                param_groups[param_name_to_group_name(param_name)]["params"].append(param)
                trainable_param_names.add(param_name)
            else:
                param.requires_grad = False

        if other_param_names is not None and type(config.trainable_param_names)!=list: 
            # print(33)
            for other_param in other_param_names:
                if re.fullmatch(other_param, param_name) and param_name not in added_param:
                    param.requires_grad = True
                    param_groups[param_name_to_group_name(param_name)]["params"].append(param)
                    trainable_param_names.add(param_name)
                # else:
                #     param.requires_grad = False

    param_groups = param_groups.values()
    if optim_name.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=config.lr)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    elif optim_name.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, eps=1e-8)
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            scale_parameter=config.scale_parameter,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return model, optimizer, trainable_param_names
