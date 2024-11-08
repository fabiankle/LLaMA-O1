# Imports and Model Initialization

import math
import random
import torch
import accelerate
import torch.nn.functional as F
from datasets import Dataset
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from grading import check
from llama_o1.constants import GENERATE_MAX_NEW_TOKENS, CUT_OFF_LEN
from llama_o1.mcts import MCTS, TreeNode
from llama_o1.policies import value_head_template, problem_declaration_template, policy_head_template
from llama_o1.utils import set_left_truncate, get_root, path_to_string, get_max_node_id_in_tree, \
    manual_seed, length_normed_log_probs, clean_generated_text


import pickle
import gzip
import numpy as np
import os


def padding_nodes(tensor, max_len):
    feature_dim = tensor.size(-1)
    pad_len = max_len - tensor.size(1)
    pad_tensor = torch.zeros(tensor.size(0), pad_len, feature_dim, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=1)

def tokenize_policy_predict(nodes,tokenizer):
    with set_left_truncate(tokenizer):
        text_for_policys = [policy_head_template(node.parent, node.index) + node.state for node in nodes]
        targets = [node.state for node in nodes]
        # with set_left_padding(tokenizer):
        inputs = tokenizer(text_for_policys, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
        target = tokenizer(targets, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
    ret = {'input_ids':inputs['input_ids'],'attention_mask':inputs['attention_mask'],'target':target['input_ids'],'target_attention_mask':target['attention_mask']}
    return ret

def forward_policy_predict(model,tokenizer,inputs):
    """
        - queries the model for the input_ids of the input
        - expectes the output to contain an "answer" of same length as the "target_ids"
        - computes CE Loss by comparing the answer logits with target logit ids

    :param model:
    :param tokenizer:
    :param inputs:
    :return:  vector of log_probs for the expected target logit ids
    """
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    target_ids = inputs["target"]
    target_mask = inputs["target_attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    # extract the "answer" logits (where expected in the outputs, of the same length as target (modulo special tokens)
    logits = outputs.logits[:,:-1,:][:, -target_ids[:,1:].shape[-1] :]
    log_probs = F.log_softmax(logits, dim=-1)
    # compute loss by checking log probs at target ids (i.e. CE loss)
    seleted_log_probs = log_probs.gather(2, target_ids[:,1:].unsqueeze(-1)).squeeze(-1)
    return seleted_log_probs

def tokenize_value_predict(node,tokenizer):
    with set_left_truncate(tokenizer):
        text_for_value = value_head_template(node) + '<positive_rating>'
        inputs = tokenizer(text_for_value, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
    inputs = {'value_' + k: v for k, v in inputs.items()}
    return inputs


def forward_value_predict(model, tokenizer, inputs):
    """
     - query the model with "value_input_ids"
      - look at expected position for evaluation token (pos or negatvie)
      - return logits at this pos for "positiv_rating" , "negative_rating") (tuple)

    :param model:
    :param tokenizer:
    :param inputs:
    :return:
    """
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    input_ids = inputs.pop("value_input_ids")
    attention_mask = inputs.pop("value_attention_mask")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    pos = attention_mask.sum(dim=1) - 1  # [batch_size]

    # 获取 "<positive_rating>" 和 "<negative_rating>" 的 token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")
    indices = torch.tensor([positive_token_id, negative_token_id], device=accelerator.device)  # [2]

    # 构建索引张量
    batch_size = logits.size(0)

    # 扩展 indices 以匹配输入 logits 的维度
    selected_logit = logits[range(batch_size), pos]  # [batch_size, num_tokens]
    selected_logit = selected_logit[:, indices]      # 提取每行中指定 token 的 logits


    return selected_logit

def get_path_reward_real(node):
    path_len = 1
    reward = 0
    while node.parent:
        path_len += 1
        reward += node.true_value_from_tree if node.true_value_from_tree is not None else node.value
        node = node.parent
    return reward / path_len

def get_path_reward_sim(node):
    path_len = 1
    reward = 0
    while node.parent:
        path_len += 1
        reward += node.original_value
        node = node.parent
    return reward / path_len


def traverse_tree(node):
    """
    Generator to traverse the entire tree from the root node
    """
    visited = set()
    nodes = [node]
    while nodes:
        current_node = nodes.pop()
        if current_node not in visited:
            visited.add(current_node)
            yield current_node
            nodes.extend(current_node.children)
        else:
            continue

def compute_gae_from_node(node, gamma=0.99, lambda_=0.95):
    """ Generalized Advantage Estimation (GAE) """
    # Backtrack to the root node and record the path
    # -> get path to root node ( starting with given node)
    path = []
    current_node = node
    while current_node.parent is not None:
        path.append(current_node)
        current_node = current_node.parent

    # 从根节点（路径起点）向下遍历到目标节点，逐步计算 GAE
    # Traverse down from the root node (the start of the path) to the target node and calculate the GAE step by step
    # klemmf: unclear  down?? ( loop goes up in the tree )
    gae = 0
    factor = 1  # 用于累乘 (gamma * lambda) 的系数

    # 从根节点开始遍历路径到指定节点
    #  Traverse the path from the root node to the specified node
    # klemmf: this is false? ( we traverse from the specified node _TO_ the root node
    for i in range(len(path) - 1):  # path[-1] 是目标节点，不需要再计算 TD 误差   # is the target node, there is no need to calculate the TD error again
        current_node = path[i]
        next_node = path[i + 1]
        next_node_reward = next_node.true_value_from_tree if next_node.true_value_from_tree is not None else next_node.value
        next_node_value = next_node.value

        # 计算 TD 误差
        # Calculate TD error
        td_error = gamma * next_node_value + (next_node_reward - current_node.value)
        # 根据 GAE 累积 TD 误差
        # Cumulative TD error according to GAE
        gae += factor * td_error
        # 更新系数，准备下一步的累积
        factor *= gamma * lambda_

    return gae



def collator_fn(batch):
    indecies = [example['indices'] for example in batch]
    weights = [example['weights'] for example in batch]
    batch = {k: pad_sequence([torch.tensor(example[k]).squeeze().unsqueeze(-1) for example in batch],True,0).squeeze() for k in batch[0].keys() if k not in ['indices', 'weights']}
    batch['indices'] = torch.tensor(indecies)
    batch['weights'] = torch.tensor(weights)  
    return batch


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = []
        self.alpha = alpha  # Prioritization factor

    def add(self, data, priority):
        """Add experience to the buffer with its priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = data
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences, with probabilities proportional to their priorities."""
        if len(self.buffer) == 0:
            return [], [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled experiences."""
        indecies_list = indices.tolist()[0]
        priorities_list = priorities.tolist()
        for idx, priority in zip(indecies_list, priorities_list):
            self.priorities[idx] = priority

    def save(self, filepath):
        """Persist the replay buffer to disk with compression and efficient storage."""
        buffer_array = np.array(self.buffer, dtype=object)  # Convert to NumPy array for storage
        priorities_array = np.array(self.priorities, dtype=np.float32)
        with gzip.open(filepath, 'wb') as f:
            pickle.dump((buffer_array, priorities_array, self.pos), f)

    def load(self, filepath):
        """Load the replay buffer from disk with decompression."""
        if os.path.exists(filepath):
            with gzip.open(filepath, 'rb') as f:
                buffer_array, priorities_array, self.pos = pickle.load(f)
                self.buffer = buffer_array.tolist()
                self.priorities = priorities_array.tolist()


class RLSPTrainer(Trainer):
    def __init__(self, envoirment, model, tokenizer, mcts, replay_buffer_capacity, args, **kwargs):
        super().__init__(
            model=model,
            processing_class=tokenizer,
            args=args,
            **kwargs
        )
        self.environment = envoirment
        self.mcts: MCTS = mcts
        self.tokenizer = tokenizer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_capacity)
        self.replay_buffer_file = 'replay_buffer.pkl'
        self.replay_buffer.load(self.replay_buffer_file)  # Load existing buffer
        self.create_optimizer_and_scheduler(9999)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    def self_play(self, initial_state):
        # cf. https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch /
        # switch to "evaluation" mode in model -> turn off gradients etc.
        self.model.eval()
        """Perform self-play to generate experiences."""
        root_node = TreeNode(state=initial_state)
        root_node = self.mcts.search(root_node)
        return root_node

    def collect_experience(self, root_node):
        """Traverse the MCTS tree to collect experiences and store them in the replay buffer."""

        # Collect training data from the tree
        for node in traverse_tree(root_node):
            if node == root_node:
                continue
            reward = node.true_value_from_tree if node.true_value_from_tree is not None else node.value
            advantage = compute_gae_from_node(node)

            policy_input = tokenize_policy_predict([node,], self.tokenizer)

            advantage_tensor = torch.tensor([advantage], dtype=torch.float32).unsqueeze(0)
            value_input = tokenize_value_predict(node, self.tokenizer)
            value_target = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

            # Store the experience with initial priority
            experience = {
                'advantage': advantage_tensor,
                'value_target': value_target,
                **policy_input,
                **value_input,
            }
            # Use absolute advantage as initial priority
            priority = abs(advantage_tensor.item())
            self.replay_buffer.add(experience, priority)

    def create_dataset_from_buffer(self, batch_size, times, beta=0.4):
        """Sample a batch from the replay buffer using PER."""
        # Prepare data for the model
        data = {
            'advantage': [],
            'value_target': [],
            'input_ids':[],
            'attention_mask':[],
            'target':[],
            'target_attention_mask':[],
            'value_input_ids':[],
            'value_attention_mask':[],
            'weights': [],
            'indices': [],  # Keep track for priority updates
        }

        for _ in range(times):
        
            samples, indices, weights = self.replay_buffer.sample(batch_size, beta)
            if len(samples) == 0:
                return None  # Not enough samples to create a batch

            for sample in samples:
                data['advantage'].append(sample.get('advantage', 0))
                data['value_target'].append(sample.get('value_target', 0))
                data['input_ids'].append(sample.get('input_ids', 0))
                data['attention_mask'].append(sample.get('attention_mask', 0))
                data['target'].append(sample.get('target', 0))
                data['target_attention_mask'].append(sample.get('target_attention_mask', 0))
                data['value_input_ids'].append(sample.get('value_input_ids', 0))
                data['value_attention_mask'].append(sample.get('value_attention_mask', 0))
                data['weights'].append(weights)
                data['indices'].append(indices)

        dataset = Dataset.from_dict(data)
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss, incorporating importance-sampling weights."""

        # Compute policy loss using PPO
        #

        # compute CE loss (meaning: "does the model give the right answer?") for both the "trained" model, as well as the base model (model without adapter)
        new_policy_log_probs = forward_policy_predict(self.model, self.tokenizer, inputs)
        with torch.no_grad():
            # base model -> peft framework? -> likely: original model without adapter (to be verified=
            old_policy_log_probs = forward_policy_predict(self.model.get_base_model(), self.tokenizer, inputs).detach()

       # at this point:          new_policy_log_probs and     old_policy_log_probs :  log probs of the expected answer tokens
        target_mask = inputs['target_attention_mask']
        advantage = inputs['advantage']
        epsilon = 0.2  # PPO clip parameter
        # i.e. ratio of token probabilities model / base_model; masked with target mask
        ratio = (new_policy_log_probs - old_policy_log_probs).exp() * target_mask[:,1:]
        # todo: klemmf What is advantage in PPO?
        # excactly as in https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations
        surr1 = ratio * advantage.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_prediction = forward_value_predict(self.model, self.tokenizer, inputs)
        value_target = inputs['value_target']

        clamp_positive_rating_prob = torch.exp(torch.clamp(
            value_target, math.log(1e-6), 0
        ))
        clamp_negative_rating_prob = 1 - clamp_positive_rating_prob
        target_probs = torch.concat(
            [clamp_positive_rating_prob.unsqueeze(-1), clamp_negative_rating_prob.unsqueeze(-1)], dim=1
        )

        value_loss = F.binary_cross_entropy_with_logits(
            value_prediction, target_probs.to(self.accelerator.device)
        )

        # Combine losses
        total_loss = policy_loss + value_loss

        if total_loss == 0:
            return total_loss

        # Apply importance-sampling weights
        weights = torch.tensor(inputs['weights'], dtype=torch.float32).to(total_loss.device)
        total_loss = total_loss * weights
        td_error = total_loss.sum(dim=-1).detach().abs().cpu().numpy()
        total_loss = total_loss.mean()
        print(f'Policy Loss: {policy_loss}, Value Loss: {value_loss}, Total Loss: {total_loss}')
        if return_outputs:
            return total_loss, td_error
        else:
            return total_loss

    def update_priorities(self, indices, td_errors):
        """Update priorities in the replay buffer based on TD-errors."""
        new_priorities = td_errors + 1e-6  # Add small epsilon to avoid zero priorities
        self.replay_buffer.update_priorities(indices, new_priorities)

    def train(self, num_iterations, beta_start=0.4, beta_frames=100000, **kwargs):
        manual_seed(os.getpid())
        frame_idx = 1
        beta = beta_start
        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")


            # Self-play to collect new experiences


            # sample an initial state Tuple [prompt , answer]
            #   - Prompt: <start_of_father_id>-1<end_of_father_id><start_of_local_id>0<end_of_local_id><start_of_thought><problem>{problem as in data}<end_of_thought>"
            #   - answer as in data (called ground truth)
            initial_state = self.environment.sample_initial_state()
            root_node = self.self_play(initial_state)
            self.collect_experience(root_node)

            # Anneal beta over time to 1.0
            beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
            frame_idx += 1

            # Ensure enough data has been collected
            if len(self.replay_buffer.buffer) < self._train_batch_size:
                continue  # Skip training until we have enough data

            # Sample a batch from the replay buffer
            train_dataset = self.create_dataset_from_buffer(self._train_batch_size, 10, beta=beta)

            if train_dataset is None:
                continue  # Not enough data to form a batch

            # Update the Trainer's dataset
            self.train_dataset = train_dataset
            self.data_collator = collator_fn
            
            # Create DataLoader
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                shuffle=True,  # PER handles sampling
                collate_fn=self.data_collator,
            )
            train_dataloader = self.accelerator.prepare(train_dataloader)
            self.accelerator.wait_for_everyone()
            # Training loop
            for step, inputs in enumerate(train_dataloader):
                self.model.train()
                inputs = self._prepare_inputs(inputs)

                # Compute loss and perform backpropagation
                loss, td_errors = self.compute_loss(self.model, inputs, return_outputs=True)
                loss = loss.mean()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accelerator.wait_for_everyone()
                # Update priorities in the replay buffer
                # For simplicity, we use the absolute value of the loss as the TD-error
                indices = inputs['indices'].cpu().numpy()
                self.update_priorities(indices, td_errors)
                self.accelerator.wait_for_everyone()

            print(f"Iteration {iteration + 1}/{num_iterations} completed.")

        # Save the replay buffer at the end of training
        self.replay_buffer.save(self.replay_buffer_file)



class Environment:
    def __init__(self, problems):
        """
        初始化环境。

        参数：
        - problems: 一个包含数学问题和答案的字典列表，每个字典包含 'problem' 和 'ground_truth' 键。
        """
        self.problems = problems
        self.num_problems = len(problems)
        self.inverse_mapping = {problem_declaration_template(problem['problem']): problem['ground_truth'] for problem in problems}

    def sample_initial_state(self) -> tuple[str, str]:
        """
        从问题列表中随机采样一个初始状态（数学问题）。

        返回：



        - initial_state: 选中的问题文本。
        - ground_truth: 该问题的正确答案，用于后续的答案验证。


        Randomly sample an initial state from the problem list (math problem).


        Returns:

        - initial_state: text of the selected question.

        - ground_truth: The correct answer to the question, to be used for subsequent answer verification.
        """
        selected_problem = random.choice(self.problems)
        initial_state = problem_declaration_template(selected_problem['problem'])
        ground_truth = selected_problem['ground_truth']
        return initial_state, ground_truth

    def is_terminal_state(self, state, ground_truth):
        """
        判断当前状态是否为终止状态（正确答案）。

        参数：
        - state: 当前状态文本。
        - ground_truth: 当前问题的正确答案。

        返回：
        - is_terminal: 布尔值，表示是否为终止状态。
        """
        # 使用 compute_rule_orm_head 函数判断
        result = self.compute_rule_orm_head(state, ground_truth)
        return result
    
    def get_ground_truth(self, node):
        return self.inverse_mapping.get(get_root(node).state)

    # 判断状态是否为正确答案的函数
    # Functions that determine if a state is the correct answer
    def compute_rule_orm_head(self, node):
        """
        使用 grading 模块的 check 函数判断状态是否为正确答案。

        参数：
        - state: 当前状态文本。
        - ground_truth: 当前问题的正确答案。

        返回：
        - result: 布尔值，表示状态是否为正确答案。
        """
        # 将状态和正确答案传入 check 函数进行比较

        try:
            ground_truth = self.inverse_mapping.get(get_root(node).state)
            result = check(ground_truth, node.state, "")
            return result
        except:
            return None



# # 初始化优化器
# optimizer = AdamW(model.parameters(), lr=1e-4)







if __name__ == "__main__":
    # 初始状态和 MCTS 参数
    num_simulations = 16
    num_candidates_per_expansion = 2
    exploration_const = 1.4
    discount_factor = 0.9
    reward_epsilon = 1e-6

    # 假设您已经定义了 TreeNode、MCTS 和 RLSPTrainer 类

    # 加载模型和 tokenizer
    model_name = "qq8933/OpenLongCoT-Base-Gemma2-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        use_cache=True
    )

    # 设置 LoRA 配置
    lora_config = LoraConfig(
        r=32,  # 低秩矩阵的秩
        lora_alpha=16,  # LoRA 的缩放系数
        target_modules=["k_proj", "q_proj", "o_proj", "v_proj", "down_proj", "gate_proj", "up_proj", ],
        # 目标模块，通常是查询和键的投影层
        lora_dropout=0.1,  # dropout 概率
        bias="none",  # 不在 LoRA 中包含偏置
    )

    # 使用 peft 将模型转换为 LoRA 微调模型
    model = get_peft_model(model, lora_config)

    print("Model successfully converted to LoRA format.")

    ds = load_dataset("openai/gsm8k", "main")['train']

    problems = [{"problem": p['question'], "ground_truth": p['answer']} for p in ds]

    # ds = load_dataset("lighteval/MATH", "all")['train']

    # problems = [{"problem": p['problem'], "ground_truth": p['solution']} for p in ds]

    environment = Environment(problems)

    # 创建 MCTS 实例
    # Creating an MCTS Instance
    mcts = MCTS(
        environment=environment,
        model=model,
        tokenizer=tokenizer,
        num_simulations=num_simulations,
        num_candidates_per_expansion=num_candidates_per_expansion,
        exploration_const=exploration_const,
        discount_factor=discount_factor,
        reward_epsilon=reward_epsilon
    )

    args = TrainingArguments(gradient_accumulation_steps=32,per_device_train_batch_size=4,output_dir='./output')

    # 创建 RLSPTrainer 实例
    # Creating an instance of RLSPTrainer
    trainer = RLSPTrainer(
        envoirment=environment,
        model=model,
        tokenizer=tokenizer,
        mcts=mcts,
        replay_buffer_capacity=100000,
        args=args,
    )

    accelerator = trainer.accelerator
    print(f"Accelerator.device {accelerator.device}")
    model = model.to(accelerator.device)

    # 设置训练轮数和批次大小
    # Setting the number of training rounds and batch size
    num_iterations = 1

    # 执行训练
    trainer.train(num_iterations=num_iterations)
    model.save_pretrained('./output')

