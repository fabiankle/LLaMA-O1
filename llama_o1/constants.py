import numpy as np

SELECT_PREFIX = ""
META_ACTION_TYPES = ["<expansion>", "<problem>", "<critic>", "<refine>", "<conclusion>"]
META_ACTION_TYPE_TO_INDEX = {meta: i for i, meta in enumerate(META_ACTION_TYPES)}

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
GENERATE_MAX_NEW_TOKENS = 64
CUT_OFF_LEN = 1024
MAX_CHILDREN_NUM = 5
DUMMY_TRANSITION_PROBS = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])


hint = '<hint> Try generate a reasonable rationale solution that can got final answer {GT}</hint>'
# hint = ''
hint_for_critics = f"<hint> Point out the potential flaws in the current solution. </hint>"
hint_for_refine = f"<hint> Try to refine the current solution for higher quality. </hint>"
hint_for_conclusion = "<hint> Try to summarize the current solution and draw a conclusion. Final answer should bracket in \\box{answer} </hint>"
hint_for_divide_and_conquer = f"<hint> Try divide the problem into smaller easier sub-problems and solve them divide-and-conquer. </hint>"
