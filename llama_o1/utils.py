import contextlib
import math
from functools import lru_cache

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

