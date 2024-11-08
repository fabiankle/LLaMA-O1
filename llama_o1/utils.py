import contextlib
import math
import torch
import torch.nn
from torch.nn import functional as F

import numpy as np

import random
import torch

from llama_o1.constants import META_ACTION_TYPE_TO_INDEX, META_ACTION_TYPES


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



@contextlib.contextmanager
def set_left_padding(tokenizer):
    # Store the original padding side
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side='left'
    # Set padding side to left
    tokenizer.padding_side = "left"
    try:
        yield tokenizer
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side

@contextlib.contextmanager
def set_left_truncate(tokenizer):
    # Store the original padding side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side='left'
    try:
        yield tokenizer
    finally:
        tokenizer.truncation_side = original_truncation_side

def value_to_rating_token(value):
    if math.exp(value) >= 0.5 and math.exp(value) <= 1:
        return "<positive_rating>"
    elif math.exp(value) < 0.5 and math.exp(value) >= 0:
        return "<negative_rating>"
    else:
        return "<unknow_rating>"


def tree_to_string(node):
    cur = f"<start_of_father_id>{node.parent.index if node.parent else -1}<end_of_father_id><start_of_local_id>{node.index}<end_of_local_id><start_of_thought>{node.state}<end_of_thought><start_of_rating>{value_to_rating_token(node.value)}<end_of_rating>"
    childs_strings = "\n".join([tree_to_string(child) for child in node.children])
    return cur + "\n" + childs_strings


def path_to_string(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    string = "\n".join(
        [
            f"<start_of_father_id>{node.parent.index if node.parent else -1}<end_of_father_id><start_of_local_id>{node.index}<end_of_local_id><start_of_thought>{node.state}<end_of_thought><start_of_rating>{value_to_rating_token(node.value)}<end_of_rating>"
            for node in path[::-1]
        ]
    )
    return string


def get_max_node_id_in_tree(node):
    if not node.parent:
        while node.parent:
            # todo klemmf: this does make sense?  / unreachable?
            node = node.parent
    max_id = node.index
    for child in node.children:
        max_id = max(max_id, get_max_node_id_in_tree(child))
    return max_id


def get_root(node):
    while node.parent:
        node = node.parent
    return node




def flatten_tree(node):
    """
    将树结构展开为列表，收集父节点、子节点和对应的值。
    """
    parents = []
    children = []
    values = []
    nodes = [node]
    while nodes:
        current_node = nodes.pop()
        current_idx = META_ACTION_TYPE_TO_INDEX[current_node.meta]
        for child in current_node.children:
            child_idx = META_ACTION_TYPE_TO_INDEX[child.meta]
            parents.append(current_idx)
            children.append(child_idx)
            values.append(np.exp(child.value))
            nodes.append(child)
    return np.array(parents), np.array(children), np.array(values)

def np_softmax(x):
    # 对矩阵的每一行进行 softmax 操作
    max_vals = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_vals)
    sum_e_x = np.sum(e_x, axis=1, keepdims=True)
    return e_x / sum_e_x



def clean_generated_text(text):
    return text[: text.find("<end_of_thought>")]




# 长度归一化的对数概率、熵和熵的方差计算
# Length-normalized log probability, entropy, and variance calculation of entropy
def length_normed_log_probs(sequence_ids, logits_tensor, attention_mask=None, return_entropy=False, return_varentropy=False):
    logits_tensor = logits_tensor[..., :-1, :].contiguous()
    sequence_ids = sequence_ids[..., 1:].contiguous()
    attention_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    selected_log_probs = log_probs.gather(2, sequence_ids.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        selected_log_probs = selected_log_probs * attention_mask

    summed_log_probs = selected_log_probs.sum(dim=1)
    length = sequence_ids.size(1) if attention_mask is None else attention_mask.sum(dim=1)
    normalized_log_probs = summed_log_probs / length

    if return_entropy or return_varentropy:
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        if attention_mask is not None:
            entropy = entropy * attention_mask
        summed_entropy = entropy.sum(dim=1)
        normalized_entropy = summed_entropy / length

    if return_varentropy:
        varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1)) ** 2, dim=-1)
        if attention_mask is not None:
            varentropy = varentropy * attention_mask
        summed_varentropy = varentropy.sum(dim=1)
        normalized_varentropy = summed_varentropy / length
        return normalized_log_probs, normalized_entropy, normalized_varentropy

    if return_entropy:
        return normalized_log_probs, normalized_entropy
    else:
        return normalized_log_probs


# 数值稳定的 softmax 函数
# Numerically stable softmax functions
def robust_softmax(logits):
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs

