import copy
import math
from functools import lru_cache
from random import random

import peft
from tqdm import tqdm
import numpy as np
from llama_o1.constants import MAX_CHILDREN_NUM, META_ACTION_TYPES, META_ACTION_TYPE_TO_INDEX
import torch

from llama_o1.utils import get_max_node_id_in_tree, flatten_tree, np_softmax, get_root
from main import find_max_reward_path, meta_compute_policy_head, compute_value_head, robust_softmax


# Tree Node Structure
class TreeNode:
    def __init__(self, state: str, parent=None, index=0):
        self.index = index  # Index of the node in the tree
        self.state: str = state  # Current state text representation
        self.parent: TreeNode | None = parent  # Parent node
        self.children: list[TreeNode] = []  # List of child nodes
        self.visits: int = 0  # Number of visits
        self.value: int = 0  # Value estimate of the current node
        self.policy = {}  # Policy probabilities for selecting child nodes
        self.policy_entropy = {}
        self.policy_varentropy = {}
        self.policy_cal_ready_texts = ""
        self.value_cal_ready_texts = ""
        self.true_value_from_tree = None
        self.leaf_type = ""
        self.rectify_visits = 0
        self.original_value = 0
        self.meta = '<problem>'

    def add_child(self, child_node: 'TreeNode'):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_path_reward(self):
        path_len = 1
        reward = 0
        node = self
        while node.parent:
            path_len += 1
            reward += node.value
            node = node.parent
        return reward / path_len

    def should_expand(self):
        if len(self.children) == 0:
            return True
        if len(self.children) < MAX_CHILDREN_NUM:  # max([child.value for child in self.children]) < self.value or
            return True
        return False

    def get_child_policy_prob(self, child):
        # 提取logit值并转换为数组
        logits = torch.tensor(list(self.policy.values()))
        prob, log_prob = robust_softmax(logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy.keys(), prob)}[child]

    def get_child_policy_entropy(self, child):
        # 提取logit值并转换为数组
        logits = torch.tensor(list(self.policy_entropy.values()))
        prob, log_prob = robust_softmax(logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy_entropy.keys(), prob)}[child]

    def get_child_policy_varentropy(self, child):
        # 提取logit值并转换为数组
        logits = torch.tensor(list(self.policy_varentropy.values()))
        prob, log_prob = robust_softmax(logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy_varentropy.keys(), prob)}[child]



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


# MCTS Search
class MCTS:
    def __init__(
            self,
            environment: 'Environment',
            model: peft.PeftModel,
            tokenizer,
            num_simulations=-1,
            num_candidates_per_expansion=2,
            exploration_const=1.414,
            discount_factor=0.9,
            reward_epsilon=1e-6,
            patient=2
    ):
        self.environment = environment
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations if num_simulations != -1 else 32
        self.exploration_const = exploration_const
        self.patient = patient
        self.discount_factor = discount_factor
        self.num_candidates = num_candidates_per_expansion
        self.reward_epsilon = reward_epsilon
        self.varentropy_lambda = 0.1

    def search(self, root_node: TreeNode):
        """
          - Run "simulate" for num_simulations times from the provided root_node;
          - for all leaves of the subtree of root_node, propagate upwards the "true_value_from_tree" via rectify_values_from_leaf
        """
        if not root_node.children:
            root_node.value = 0

        for _ in tqdm(range(self.num_simulations)):
            self.simulate(root_node)

            # only used for logging purposes!
            max_reward, path_len = find_max_reward_path(root_node)
            print(f'find max reward path: {max_reward} with {path_len} steps.')
            if self.patient <= 0:
                break

        for leaf in self.identify_leaf(root_node):
            if leaf.leaf_type == "successful":
                self.rectify_values_from_leaf(leaf, 0)
            else:
                self.rectify_values_from_leaf(leaf, np.log(self.reward_epsilon))

        return root_node

        # return self.get_policy_from_visits(root_node)

    def simulate(self, node: TreeNode):
        """
          Either expands and evaluates the node itself (if not children or max children not yet reached)
          , or the best of its children (in that case: recursive call!); updates the node's value and returns it

        """
        if node.is_leaf() or node.should_expand():
            value = self.expand_node(node) * self.discount_factor
        else:
            best_child = self.select_action(node)
            value = self.simulate(best_child) * self.discount_factor
        node.visits += 1
        node.value += (value - node.value) / node.visits
        return node.value

    def expand_node(self, node: TreeNode):
        """
          node expansion:



        """
        texts, policy_probs, entropys, varentropys, metas = meta_compute_policy_head(self.model, self.tokenizer, node,
                                                                                     self.num_candidates,
                                                                                     envoirment=self.environment)

        for i, (text, policy_prob, entropy, varentropy, meta) in enumerate(
                zip(texts, policy_probs, entropys, varentropys, metas)):
            child_node = TreeNode(
                state=text, parent=node, index=get_max_node_id_in_tree(node) + 1
            )
            # child_node.policy = policy_probs[i]
            node.policy[child_node] = policy_prob
            node.policy_entropy[child_node] = entropy
            node.policy_varentropy[child_node] = varentropy
            node.add_child(child_node)
            child_node.value = self.compute_value(child_node)
            child_node.meta = meta
            # if child_node.meta == "<conclusion>":
            orm = self.environment.compute_rule_orm_head(child_node)
            if orm == True:
                self.patient -= 1
                child_node.leaf_type = "successful"
            elif orm == False:
                child_node.leaf_type = "failed"
            print(
                f"Id:{node.index}->{child_node.index}, Child: {text}, Policy: {node.get_child_policy_prob(child_node)}, Value: {math.exp(child_node.value)}"
            )
        return self.select_action(node).value

    def compute_value(self, node):
        # Use the model to predict the value of the current state
        value = compute_value_head(self.model, self.tokenizer, node)
        node.value = value
        node.original_value = copy.deepcopy(value)
        return value

    def select_action(self, node):
        total_visits = sum(child.visits for child in node.children)
        ucb_scores = [
            (
                    child.value
                    + self.exploration_const
                    * node.get_child_policy_prob(child)
                    # * node.get_child_policy_entropy(child)
                    * np.sqrt(total_visits)
                    / (1 + child.visits)
                    + self.varentropy_lambda * node.get_child_policy_varentropy(child)
            ) * random.uniform(0.8, 1.2)
            for child in node.children
        ]
        return node.children[np.argmax(ucb_scores)]

    def identify_leaf(self, node: TreeNode):
        result = set()
        if node.is_leaf() or node.leaf_type in ["successful", "failed"]:
            result.add(node)
        else:
            for child in node.children:
                result |= self.identify_leaf(child)
        return result

    def rectify_values_from_leaf(self, node: TreeNode, value: float):
        """
            set and propagate the true_value_from_tree of the node rescursively upwards
            - value is (1- 1/node.rectify_visits) * old_value + 1/node.rectify_visits new_value
            - upward propagation contains discount_factor

        """
        node.rectify_visits += 1

        if not node.true_value_from_tree:
            node.true_value_from_tree = value
        else:
            node.true_value_from_tree += (
                                                 value - node.true_value_from_tree
                                         ) / node.rectify_visits
        if node.parent:
            self.rectify_values_from_leaf(
                node.parent, node.true_value_from_tree * self.discount_factor
            )
