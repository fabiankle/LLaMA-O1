import numpy as np


SELECT_PREFIX = ""
META_ACTION_TYPES = ["<expansion>", "<problem>", "<critic>", "<refine>", "<conclusion>"]
META_ACTION_TYPE_TO_INDEX = {meta: i for i, meta in enumerate(META_ACTION_TYPES)}

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
GENERATE_MAX_NEW_TOKENS = 64
CUT_OFF_LEN = 1024
MAX_CHILDREN_NUM = 5
DUMMY_TRANSITION_PROBS = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

