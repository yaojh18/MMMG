I_OBJECT_COUNT_PROMPT = lambda obj: f"How many {obj} are there in the given image? Choose from the options:\nA. Less than 3\nB. 3\nC. 4\nD. 5\nE. 6\nF. More than 6\nRespond only with the option letter (A, B, C, D, E or F). Do not provide any explanation, reasoning, or additional information."

I_OBJECT_EXIST_PROMPT = lambda obj: f"Is/Are there {obj} in the given image? Answer only yes or no.\n"

I_OBJECT_EXIST_COT_PROMPT = lambda obj: f"Is/Are there {obj} in the given image? Explain step by step and end your answer with \"Yes\" or \"No\".\n"

I_SPACIAL_ABSOLUTE_PROMPT = lambda obj: f"The {obj} is located in which section of the image? Choose from the options:\nA. bottom left B. bottom right C. up left D. up right E. none of the above (positioned in a more central way)\nExplain step by step and end your answer with \"Answer: [only an optional letter].\"\n"

I_SPACIAL_RELATIVE_LR = lambda obj1, obj2: f"Looking at the 2D composition of the image, what is the horizontal alignment relationship between the {obj1} and the {obj2}? Choose from the options:\nA. the {obj1} is obviously to the left of the {obj2}.\nB. the {obj1} is obviously to the right of the {obj2}.\nC. the {obj1} is neither obviously to the right nor left of the {obj2}.\nExplain step by step and end your answer with \"Answer: [only an optional letter].\"\n"

I_SPACIAL_RELATIVE_UD = lambda obj1, obj2: f"Looking at the 2D composition of the image, what is the vertical alignment relationship between the {obj1} and the {obj2}? Choose from the options:\nA. the {obj1} is obviously positioned higher than the {obj2}.\nB. the {obj1} is obviously positioned lower than the {obj2}.\nC. the {obj1} is neither obviously positioned higher nor lower than the {obj2}.\nExplain step by step and end your answer with \"Answer: [only an optional letter].\"\n"

I_OCR_ENGLISH_PROMPT = lambda obj: f"### Instruction:\nRecognize all the major texts (ignore small texts on the edge) ONLY on {obj}. Only recognize texts in Latin alphabet characters (a-z, A-Z). Do not correct the text if it is misspelled, nonsense or wrong, output the most direct recognition result. Do not call any function.\n### Output format:\nOutput an executable Python list of all recognized texts from top to down, from left to right, e.g. [\"Hello World\", \"Good morning\"]. Output an empty list if the there is no text on {obj}."

I_OCR_CHINESE_PROMPT = "### Instruction:\nYou are a conservative text recognition model. Your task is to recognize all the major Chinese characters in the given image. If the Chinese characters in the image are wrongly written or distorted, you should return empty result. Do not call any function.\n### Output format:\nOny a string of all recognized texts from top to down, from left to right. Do not add quotations."

with open('./prompts/i_agent.txt', 'r', encoding='utf-8') as f:
    I_AGENT_PROMPT = ''.join(f.readlines())

with open('./prompts/a_agent.txt', 'r', encoding='utf-8') as f:
    A_AGENT_PROMPT = ''.join(f.readlines())
