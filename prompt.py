I_OBJECT_COUNT_PROMPT = lambda obj: f"How many {obj} are there in the given image? Choose from the options:\nA. Less than 3\nB. 3\nC. 4\nD. 5\nE. 6\nF. More than 6\nRespond only with the option letter (A, B, C, D, E or F). Do not provide any explanation, reasoning, or additional information."

I_OBJECT_EXIST_PROMPT = lambda obj: f"Is/Are there {obj} in the given image? Answer only yes or no.\n"

I_SPACIAL_ABSOLUTE_PROMPT = lambda obj: f"Where is {obj} in the given image? Choose from the options:\nA. bottom left B. bottom right C. up left D. up right E. none of the locations above F. multiple objects or no object exist\nDo not provide any explanation, reasoning, or additional information.\n"

I_SPACIAL_RELATIVE_FR = lambda obj1, obj2: f"What is the relative left-right relationship of {obj1} and {obj2} in the given image? Be careful there may be multimple objects. Choose from the options:\nA. {obj1} is to the left of {obj2}.\nB. {obj1} is to the right of {obj2}.\nC. {obj1} is either distinctly to the left or right of the {obj2}.\nD. multiple {obj1} or {obj2} are in the given image and their relationship are inconsistent, thus unable to determine.\nE. either {obj1} or {obj2} is not clearly visible in the given image.\nDo not provide any explanation, reasoning, or additional information. Do not consider perspective.\n"

I_SPACIAL_RELATIVE_UD = lambda obj1, obj2: f"What is the relative up-down relationship of {obj1} and {obj2} in the given image? Be careful there may be multimple objects. Choose from the options:\nA. {obj1} is above {obj2}.\nB. {obj1} is below {obj2}.\nC. {obj1} is either distinctly above or below {obj2}.\nD. multiple {obj1} or {obj2} are in the given image and their relationship are inconsistent, thus unable to determine.\nE. either {obj1} or {obj2} is not clearly visible in the given image.\nDo not provide any explanation, reasoning, or additional information. Do not consider perspective.\n"

I_OCR_ENGLISH_PROMPT = lambda obj: f"### Instruction:\nRecognize all the major texts (ignore small texts on the edge) on {obj}. Only recognize texts in Latin alphabet characters (a-z, A-Z). Do not correct the text if it is misspelled, nonsense or wrong, output the most direct recognition result. Do not call any function.\n### Output format:\nOutput an executable Python list of all recognized texts from top to down, from left to right, e.g. [\"Hello World\", \"Good morning\"]. Output an empty list if the there is no text on {obj}."

I_OCR_CHINESE_PROMPT = "### Instruction:\nYou are a conservative text recognition model. Your task is to recognize all the major Chinese characters in the given image. If the Chinese characters in the image are wrongly written or distorted, you should return empty result. Do not call any function.\n### Output format:\nOny a string of all recognized texts from top to down, from left to right. Do not add quotations."

with open('./prompts/i_agent.txt', 'r', encoding='utf-8') as f:
    I_AGENT_PROMPT = ''.join(f.readlines())

with open('./prompts/a_agent.txt', 'r', encoding='utf-8') as f:
    A_AGENT_PROMPT = ''.join(f.readlines())
