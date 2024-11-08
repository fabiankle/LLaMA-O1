from functools import lru_cache

import accelerate
import torch
import torch.nn
import numpy as np
from llama_o1.constants import hint_for_critics, \
    hint_for_divide_and_conquer, hint_for_refine, hint, CUT_OFF_LEN, GENERATE_MAX_NEW_TOKENS, META_ACTION_TYPES, \
    META_ACTION_TYPE_TO_INDEX
from llama_o1.utils import get_max_node_id_in_tree, set_left_truncate, length_normed_log_probs, clean_generated_text, \
    robust_softmax, path_to_string, get_root, flatten_tree, np_softmax


def cal_meta_transition_probs(node: 'TreeNode'):
    num_meta_actions = len(META_ACTION_TYPES)
    # 展开树结构，获取父节点索引、子节点索引和对应的值
    # Expand the tree structure and get the parent node index, child node index and corresponding values
    parents, children, values = flatten_tree(node)
    # 初始化转移概率矩阵
    # Initialize the transfer probability matrix
    TransitionProbs = np.zeros((num_meta_actions, num_meta_actions))
    # 使用 NumPy 的高级索引和累加来更新矩阵
    # Use NumPy's advanced indexing and accumulation to update matrices
    if len(parents) > 0:
        np.add.at(TransitionProbs, (parents, children), values)
    return TransitionProbs



@lru_cache()
def sampling_meta_action(node: 'TreeNode', num=1, TransitionProbs=None):
    if TransitionProbs is None:
        root = get_root(node)
        TransitionProbs = cal_meta_transition_probs(root)
    # Calculate softmax for transfer probability
    transition_probs_softmax = np_softmax(TransitionProbs)
    i = META_ACTION_TYPE_TO_INDEX[node.meta]
    p = transition_probs_softmax[i]
    # 进行采样
    meta_actions = np.random.choice(META_ACTION_TYPES, size=num, p=p)
    return meta_actions


# 模板生成函数
# Template generation functions
def problem_declaration_template(problem):
    return f"<start_of_father_id>-1<end_of_father_id><start_of_local_id>0<end_of_local_id><start_of_thought><problem>{problem}<end_of_thought>"

def selection_head_template(tree):
    return tree.to_string() + "\n<start_of_father_id>"

def policy_head_template(selected_node, local_id, meta="", hint=""):
    return (
        path_to_string(selected_node)
        + f"{hint}\n<start_of_father_id>{selected_node.index if selected_node else -1}<end_of_father_id><start_of_local_id>{local_id}<end_of_local_id><start_of_thought>{meta}"
    )

def value_head_template(selected_node):
    return (
        path_to_string(selected_node.parent)
        + f"\n<start_of_father_id>{selected_node.parent.index if selected_node.parent else -1}<end_of_father_id><start_of_local_id>{selected_node.index}<end_of_local_id><start_of_thought>{selected_node.state}<end_of_thought><start_of_rating>"
    )

selection_head_stopping_criteria = ["<end_of_father_id>"]

policy_head_stopping_criteria = ["<end_of_thought>"]

value_head_stopping_criteria = ["<end_of_rating>"]

accelerator = accelerate.Accelerator()

# 策略生成的主要函数
# Value header generation function
@torch.no_grad()
def compute_policy_head(model, tokenizer, selected_node, num_candidates=3, meta="", envoirment=None):
    local_id = get_max_node_id_in_tree(selected_node) + 1
    hint_text = {
        "<conclusion>": hint_for_critics,
        "<problem>": hint_for_divide_and_conquer,
        "<critic>": hint_for_critics,
        "<refine>": hint_for_refine,
    }.get(meta, hint.format(GT=envoirment.get_ground_truth(selected_node)))

    inputs_string = policy_head_template(selected_node, local_id, meta, hint_text)
    with set_left_truncate(tokenizer):
        inputs = tokenizer(
            inputs_string,
            return_tensors="pt",
            truncation=True,
            padding='longest',
            max_length=CUT_OFF_LEN
        )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    outputs = accelerator.unwrap_model(model).generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=GENERATE_MAX_NEW_TOKENS,
        do_sample=True,
        num_return_sequences=num_candidates,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=1.5,
        output_logits=True,
        stop_strings=policy_head_stopping_criteria,
        tokenizer=tokenizer,
    )

    generated_sequences = outputs.sequences[:, inputs['input_ids'].size(1):]
    generated_sequences_mask = generated_sequences != tokenizer.pad_token_id
    generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    logits = torch.stack(outputs.logits, dim=1)
    normalized_log_probs, normalized_entropy, varentropy = length_normed_log_probs(
        generated_sequences, logits, attention_mask=generated_sequences_mask, return_entropy=True, return_varentropy=True
    )

    normalized_probs = torch.exp(normalized_log_probs)

    generated_texts = [meta + clean_generated_text(text) for text in generated_texts]
    for i, generated_text in enumerate(generated_texts):
        if not generated_text.startswith(meta):
            generated_texts[i] = meta + generated_text

    return generated_texts, normalized_probs.tolist(), normalized_entropy.tolist(), varentropy.tolist(), [meta,] * num_candidates



# 价值头生成函数
@torch.no_grad()
def compute_value_head(model, tokenizer, node):
    text_for_value = value_head_template(node) + '<positive_rating>'
    with set_left_truncate(tokenizer):
        inputs = tokenizer(text_for_value, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits

    last_logits = logits[:, -2, :]
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")

    positive_logit = last_logits[:, positive_token_id]
    negative_logit = last_logits[:, negative_token_id]
    value_logits = torch.stack([positive_logit, negative_logit], dim=1)

    probs, log_probs = robust_softmax(value_logits)
    return log_probs[:, 0].item()


# 元策略生成函数
# Meta-strategy generation function
@torch.no_grad()
def meta_compute_policy_head(model, tokenizer, selected_node, num_candidates=3, meta_ratio=0.5, envoirment=None):
    metas = sampling_meta_action(selected_node, num_candidates)
    generated_texts, policy_probs, normalized_entropys, varentropys = [], [], [], []

    for meta in metas:
        texts, policy_probs, normalized_entropy, varentropy, _ = compute_policy_head(model, tokenizer,
            selected_node, num_candidates=1, meta=meta, envoirment=envoirment
        )
        generated_texts.append(texts[0])
        policy_probs.append(policy_probs[0])
        normalized_entropys.append(normalized_entropy[0])
        varentropys.append(varentropy[0])

    return generated_texts, policy_probs, normalized_entropys, varentropys, metas
