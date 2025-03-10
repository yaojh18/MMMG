"""
TODO: migrate all the prompts into this file.
"""
I_OBJECT_COUNT_PROMPT = lambda obj: (f"How many {obj} are there in the given image? Choose from the options:\n"
                                      f"A. Less than 3\nB. 3\nC. 4\nD. 5\nE. 6\nF. More than 6\n"
                                      f"Respond only with the option letter (A, B, C, D, E or F). Do not provide any "
                                      f"explanation, reasoning, or additional information.")

I_OBJECT_EXIST_PROMPT = lambda obj: f"Is/Are there {obj} in the given image? Answer only yes or no.\n"

with open('./prompts/i_agent.txt', 'r', encoding='utf-8') as f:
    I_AGENT_PROMPT = ''.join(f.readlines())