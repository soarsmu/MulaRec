import os
import random
import numpy as np
import pandas as pd
import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 src_ant,
                 src_code,
                 sim_post,
                 target,
                 ):
        self.idx = idx
        self.src_ant = src_ant
        self.src_code = src_code
        self.sim_post = sim_post
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    
    df = pd.read_csv(filename)
    df = df.fillna("")
    code = df['source_code'].astype("string").tolist()
    ant = df['annotation'].astype("string").tolist()
    post = df['related_so_question'].astype("string").tolist()
    # post = df['similar_post'].astype("string").tolist()
    api_seq = df['target_api'].astype("string").tolist()
    
    for i in range(len(code)):
        examples.append(
            Example(
                idx=i,
                src_code=code[i].lower(),
                # src_ant=ant[i].lower(),
                src_ant="",
                sim_post=post[i].lower(),
                target=api_seq[i].lower(),
            )
        )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 src_code_ids,
                 src_ant_ids,
                 sim_post_ids,
                 target_ids,
                 src_code_mask,
                 src_ant_mask,
                 sim_post_mask,
                 target_mask,
                 ):
        self.example_id = example_id
        self.src_code_ids = src_code_ids
        self.src_ant_ids = src_ant_ids
        self.sim_post_ids = sim_post_ids
        self.target_ids = target_ids
        self.src_code_mask = src_code_mask
        self.src_ant_mask = src_ant_mask
        self.sim_post_mask = sim_post_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length,\
    max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, desc='convert examples to features...')):
        # source code
        src_code_tokens = tokenizer.tokenize(example.src_code)[:max_source_length - 2]
        src_code_tokens = [tokenizer.cls_token] + src_code_tokens + [tokenizer.sep_token]
        src_code_ids = tokenizer.convert_tokens_to_ids(src_code_tokens)
        src_code_mask = [1] * (len(src_code_tokens))
        padding_length = max_source_length - len(src_code_ids)
        src_code_ids += [tokenizer.pad_token_id] * padding_length
        src_code_mask += [0] * padding_length
        
        # annotation
        src_ant_tokens = tokenizer.tokenize(example.src_ant)[:max_source_length - 2]
        src_ant_tokens = [tokenizer.cls_token] + src_ant_tokens + [tokenizer.sep_token]
        src_ant_ids = tokenizer.convert_tokens_to_ids(src_ant_tokens)
        src_ant_mask = [1] * (len(src_ant_tokens))
        padding_length = max_source_length - len(src_ant_ids)
        src_ant_ids += [tokenizer.pad_token_id] * padding_length
        src_ant_mask += [0] * padding_length

        # similar post
        sim_post_tokens = tokenizer.tokenize(example.sim_post)[:max_source_length - 2]
        sim_post_tokens = [tokenizer.cls_token] + sim_post_tokens + [tokenizer.sep_token]
        sim_post_ids = tokenizer.convert_tokens_to_ids(sim_post_tokens)
        sim_post_mask = [1] * (len(sim_post_tokens))
        padding_length = max_source_length - len(sim_post_ids)
        sim_post_ids += [tokenizer.pad_token_id] * padding_length
        sim_post_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
            
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                src_code_ids,
                src_ant_ids,
                sim_post_ids,
                target_ids,
                src_code_mask,
                src_ant_mask,
                sim_post_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def set_logger(log_path):
    """ e.g., logging.info """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt = '%F %A %T'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)