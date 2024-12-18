import numpy as np
from sklearn.metrics import cohen_kappa_score
import evaluate
from typing import Callable
import Levenshtein

from model import *
from interface import *


class EvalUnit:
    inst_name: str

    def __init__(self, model_name: str, sample_size=4):
        self.inst_list = []
        self.model_name = model_name
        self.sample_size = sample_size
        with open(f'./seed_instruction/{self.inst_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                self.inst_list.append(json.loads(line.strip()))

        if os.path.exists(f'./output/{model_name}/{self.inst_name}.jsonl'):
            self.res_list = []
            with open(f'./output/{model_name}/{self.inst_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.res_list.append(json.loads(line.strip()))
            for res in self.res_list:
                image_list = []
                for image_name in res['image_list']:
                    image_list.append(Image.open(f'./output/{model_name}/image/{self.inst_name}_{image_name}.png'))
                res['image_list'] = image_list
                audio_list = []
                for audio_name in res['audio_list']:
                    audio_list.append(sf.read(f'./output/{model_name}/audio/{self.inst_name}_{audio_name}.flac'))
                res['audio_list'] = audio_list
            if len(self.inst_list) * sample_size == len(self.res_list):
                return

        model = eval(f'{model_name}()')
        query_list = [inst['instruction'] for inst in self.inst_list for _ in range(sample_size)]
        self.res_list = model.generate(query_list)
        self.save(save_all=True)

    def save(self, save_all=False):
        output_path = f'./output/{self.model_name}/'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path + 'image/', exist_ok=True)
        os.makedirs(output_path + 'audio/', exist_ok=True)
        image_list = []
        audio_list = []
        output_list = []
        image_idx = 0
        audio_idx = 0
        for res in self.res_list:
            output = res.copy()
            output['image_list'] = list(range(image_idx, image_idx + len(res['image_list'])))
            output['audio_list'] = list(range(audio_idx, audio_idx + len(res['audio_list'])))
            image_idx += len(res['image_list'])
            audio_idx += len(res['audio_list'])
            image_list += res['image_list']
            audio_list += res['audio_list']
            output_list.append(output)
        with open(output_path + f'{self.inst_name}.jsonl', 'w', encoding='utf-8') as file:
            for data in output_list:
                file.write(json.dumps(data) + '\n')
        if save_all:
            for idx, image in enumerate(image_list):
                image.save(output_path + f'image/{self.inst_name}_{idx}.png')
        if save_all:
            for idx, audio in enumerate(audio_list):
                sf.write(output_path + f'audio/{self.inst_name}_{idx}.flac', audio, SAMPLE_RATE)

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def calculate_metrics(self):
        pass


class IObject(EvalUnit):
    label_list: tuple

    @staticmethod
    @abstractmethod
    def instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def human_instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def gpt_judge_process_func(res: str):
        pass

    @staticmethod
    @abstractmethod
    def human_judge_process_func(res: str):
        pass

    def evaluate(self):
        eval_inst_list = [inst for inst in self.inst_list for _ in range(self.sample_size)]

        if not all(['gpt_eval' in res for res in self.res_list]):
            queries = []
            for data, inst in zip(self.res_list, eval_inst_list):
                queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + self.instruction_func(inst), images=data['image_list']))
            responses = batch(query_openai, queries, model='gpt-4o', temperature=0.0)
            parsed_responses = [self.gpt_judge_process_func(res) for res in responses]
            for data, gpt_eval in zip(self.res_list, parsed_responses):
                data['gpt_eval'] = gpt_eval
            self.save()

        if not all(['human_eval' in res for res in self.res_list]):
            interface = MultiLabelInterface(
                label_list=self.label_list,
                eval_inst_list=eval_inst_list,
                data_list=self.res_list
            )
            interface.start()
            for data, human_eval in zip(self.res_list, interface.eval_list):
                data['human_eval'] = self.human_judge_process_func(human_eval)
            self.save()

        if self.inst_name == 'i_object_counting':
            for data, inst in zip(self.res_list, eval_inst_list):
                data['human_eval'] = float(data['human_eval'] == (inst['count'] - 2))
                data['gpt_eval'] = float(data['gpt_eval'] == (inst['count'] - 2))
            self.save()

    def calculate_metrics(self):
        gpt_eval_list = [res['gpt_eval'] for res in self.res_list]
        human_eval_list = [res['human_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy for {self.inst_name}: ", np.mean(gpt_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(
            f"Cohen's Kappa for {self.inst_name}: ",
            1.0 if gpt_eval_list == human_eval_list else cohen_kappa_score(gpt_eval_list, human_eval_list)
        )
        print(
            f"Pearson Correlation for {self.inst_name}: ",
            1.0 if gpt_eval_list == human_eval_list else np.corrcoef(gpt_eval_list, human_eval_list)[0, 1]
        )


class IObjectInclude(IObject):
    inst_name = 'i_object_include'
    label_list = ("Yes", "No")

    @staticmethod
    def instruction_func(inst: dict):
        return f"Is/Are there {inst['object']} in the given image? Answer only yes or no.\n"

    @staticmethod
    def human_instruction_func(inst: dict):
        return f"Is/Are there {inst['object']} in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if res.lower().startswith('yes') else 0.0

    @staticmethod
    def human_judge_process_func(res: str):
        return 1.0 if res == 0 else 0.0


class IObjectExclude(IObject):
    inst_name = 'i_object_exclude'
    label_list = ("Yes", "No")

    @staticmethod
    def instruction_func(inst: dict):
        return f"Is/Are there {inst['object']} in the given image? Answer only yes or no.\n"

    @staticmethod
    def human_instruction_func(inst: dict):
        return f"Is/Are there {inst['object']} in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if res.lower().startswith('no') else 0.0

    @staticmethod
    def human_judge_process_func(res: str):
        return 1.0 if res == 1 else 0.0


class IObjectCoT(IObject):
    inst_name = 'i_object_cot'
    label_list = ("Yes", "No")

    @staticmethod
    def instruction_func(inst: dict):
        return f"Is the given image about {inst['object']}? Answer only yes or no.\n"

    @staticmethod
    def human_instruction_func(inst: dict):
        return f"Is the given image about {inst['object']}?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if res.lower().startswith('yes') else 0.0

    @staticmethod
    def human_judge_process_func(res: str):
        return 1.0 if res == 0 else 0.0


class IObjectCounting(IObject):
    inst_name = 'i_object_counting'
    label_list = ("A. Less than 3", "B. 3", "C. 4", "D. 5", "E. 6", "F. More than 6")

    @staticmethod
    def instruction_func(inst: dict):
        return (f"How many {inst['object']} are there in the given image? Choose from the options:\n"
                f"A. Less than 3\nB. 3\nC. 4\nD. 5\nE. 6\nF. More than 6\n"
                f"Respond only with the option letter (A, B, C, D, E or F). Do not provide any explanation, reasoning, or additional information.")

    @staticmethod
    def human_instruction_func(inst: dict):
        return f"How many {inst['object']} are there in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd', 'e', 'f') else FAILED_TOKEN

    @staticmethod
    def human_judge_process_func(res: str):
        return res


class ISpacial(EvalUnit):
    @staticmethod
    @abstractmethod
    def instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def human_instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def gpt_judge_parse_func(res: str):
        pass

    @staticmethod
    @abstractmethod
    def gpt_judge_process_func(res: str, const: tuple):
        pass

    @staticmethod
    @abstractmethod
    def human_judge_process_func(res: str):
        pass

    def evaluate(self):
        eval_inst_list = [inst for inst in self.inst_list for _ in range(self.sample_size)]

        if not all(['gpt_eval' in res for res in self.res_list]):
            queries = []
            idx = 0
            for data, inst in zip(self.res_list, eval_inst_list):
                for constraint in inst['constraints']:
                    queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + self.instruction_func(constraint),
                                                        images=data['image_list']))
                data['gpt_eval'] = list(range(idx, idx + len(inst['constraints'])))
                idx += len(inst['constraints'])
            responses = batch(query_openai, queries, model='gpt-4o', temperature=0.0)
            parsed_responses = [self.gpt_judge_parse_func(res) for res in responses]
            for data, inst in zip(self.res_list, eval_inst_list):
                data['gpt_eval'] = [self.gpt_judge_process_func(parsed_responses[gpt_eval], constraint) for
                                    gpt_eval, constraint in zip(data['gpt_eval'], inst['constraints'])]
            self.save()

        if not all(['human_eval' in res for res in self.res_list]):
            human_queries = []
            human_res_list = []
            idx = 0
            for data, inst in zip(self.res_list, eval_inst_list):
                human_res_list += [data] * len(inst['constraints'])
                for constraint in inst['constraints']:
                    human_queries.append(self.human_instruction_func(constraint))
                data['human_eval'] = list(range(idx, idx + len(inst['constraints'])))
                idx += len(inst['constraints'])
            interface = MultiLabelInterface(
                label_list=("Yes", "No"),
                eval_inst_list=human_queries,
                data_list=human_res_list
            )
            interface.start()
            for data in self.res_list:
                data['human_eval'] = [self.human_judge_process_func(interface.eval_list[idx]) for idx in data['human_eval']]
            self.save()

    def calculate_metrics(self):
        gpt_eval_list = [np.mean(res['gpt_eval']) for res in self.res_list]
        human_eval_list = [np.mean(res['human_eval']) for res in self.res_list]
        print(f"Auto evaluation accuracy for {self.inst_name}: ", np.mean(gpt_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        gpt_eval_list = np.concatenate([res['gpt_eval'] for res in self.res_list])
        human_eval_list = np.concatenate([res['human_eval'] for res in self.res_list])
        print(f"Cohen's Kappa for {self.inst_name}: ", cohen_kappa_score(gpt_eval_list, human_eval_list))
        print(f"Pearson Correlation for {self.inst_name}: ", np.corrcoef(gpt_eval_list, human_eval_list)[0, 1])


class ISpacialAbsolute(ISpacial):
    inst_name = 'i_spacial_absolute'

    @staticmethod
    def instruction_func(const: dict):
        return (f'Where is {const[0]} in the given image? Choose from the options:\n'
                f'A. bottom left B. bottom right C. up left D. up right E. none of above F. object not exist\n'
                f'Do not provide any explanation, reasoning, or additional information.\n')

    @staticmethod
    def human_instruction_func(const: dict):
        return f'Is there {const[0]} in the given image and locate at the {const[1]} part of the image?\n'

    @staticmethod
    def gpt_judge_parse_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd', 'e', 'f') else FAILED_TOKEN

    @staticmethod
    def gpt_judge_process_func(res: str, const: tuple):
        return float(res == {'bottom left': 0, 'bottom right': 1, 'top left': 2, 'top right': 3}[const[1]])

    @staticmethod
    def human_judge_process_func(res: str):
        return float(res == 0)


class ISpacialRelative(ISpacial):
    inst_name = 'i_spacial_relative'

    @staticmethod
    def instruction_func(const: dict):
        if const[2] in ('to the left of', 'to the right of'):
            return (
                f'What is the relative left-right relationship of {const[0]} and {const[1]} in the given image? Be careful there may be multimple objects. Choose from the options:\n'
                f'A. {const[0]} is to the left of {const[1]}.\n'
                f'B. {const[0]} is to the right of {const[1]}.\n'
                f'C. {const[0]} is either distinctly to the left or right of the {const[1]}.\n'
                f'D. multiple {const[0]} or {const[1]} are in the given image and their relationship are inconsistent, thus unable to determine.\n'
                f'E. either {const[0]} or {const[1]} is not clearly visible in the given image.\n'
                f'Do not provide any explanation, reasoning, or additional information. Do not consider perspective.\n')
        else:
            return (
                f'What is the relative up-down relationship of {const[0]} and {const[1]} in the given image? Be careful there may be multimple objects. Choose from the options:\n'
                f'A. {const[0]} is above {const[1]}.\n'
                f'B. {const[0]} is below {const[1]}.\n'
                f'C. {const[0]} is either distinctly above or below {const[1]}.\n'
                f'D. multiple {const[0]} or {const[1]} are in the given image and their relationship are inconsistent, thus unable to determine.\n'
                f'E. either {const[0]} or {const[1]} is not clearly visible in the given image.\n'
                f'Do not provide any explanation, reasoning, or additional information. Do not consider perspective.\n')

    @staticmethod
    def human_instruction_func(const: dict):
        return f'Is {const[0]} {const[2]} {const[1]} in the given image?\n'

    @staticmethod
    def gpt_judge_parse_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd') else FAILED_TOKEN

    @staticmethod
    def gpt_judge_process_func(res: str, const: tuple):
        if const[2] in ('to the left of', 'above'):
            return float(res == 0)
        else:
            return float(res == 1)

    @staticmethod
    def human_judge_process_func(res: str):
        return float(res == 0)


class IOCR(EvalUnit):
    inst_name = 'i_ocr'   # or 'i_ocr_long'

    def __init__(self, inst_name, **kwargs):
        self.inst_list = inst_name
        super().__init__(**kwargs)

    def evaluate(self):
        if not all(['gpt_eval' in res for res in self.res_list]):
            eval_inst_list = [inst for inst in self.inst_list for _ in range(self.sample_size)]
            instruction = ("### Instruction:\n"
                           "Recognize all the major English texts in the given image. Do not correct the text if it is misspelled, nonsense or wrong, output the most direct recognition result. Do not call any function.\n"
                           "### Output format:\n"
                           "[only a executable Python list of all recognized texts from top to down, from left to right]")
            queries = []
            for data, inst in zip(self.res_list, eval_inst_list):
                queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + instruction, images=data['image_list']))
            responses = batch(query_openai, queries, model='gpt-4o', temperature=0.0)
            for data, res, inst in zip(self.res_list, responses, eval_inst_list):
                try:
                    data['gpt_eval'] = ' '.join(eval(res)).lower().strip()
                except Exception:
                    data['gpt_eval'] = ''
            self.save()

        if not all(['human_eval' in res for res in self.res_list]):
            interface = FreeLabelInterface(
                eval_inst_list=["Please type the major recognized texts (case insensitive) from top to down, from left to right in the given image."] * len(self.res_list),
                data_list=self.res_list
            )
            interface.start()
            for data, human_eval in zip(self.res_list, interface.eval_list):
                data['human_eval'] = human_eval.lower().strip()
            self.save()

    def calculate_metrics(self):
        label_list = [inst['text'].lower().strip() for inst in self.inst_list for _ in range(self.sample_size)]
        label_list = [label for label, res in zip(label_list, self.res_list) if res['gpt_eval'] != '']
        gpt_eval_list = [res['gpt_eval'] for res in self.res_list if res['gpt_eval'] != '']
        human_eval_list = [res['human_eval'] for res in self.res_list if res['gpt_eval'] != '']
        rouge = evaluate.load('rouge')

        rouge_list = [rouge.compute(predictions=[gpt_eval], references=[human_eval])['rougeL']
                      for gpt_eval, human_eval in zip(gpt_eval_list, human_eval_list)]
        dist_list = [1.0 - Levenshtein.distance(gpt_eval, human_eval) / max(len(gpt_eval), len(human_eval))
                     for gpt_eval, human_eval in zip(gpt_eval_list, human_eval_list)]
        print(f"RougeL for {self.inst_name}: ", np.mean(rouge_list))
        print(f"Edit distance for {self.inst_name}: ", np.mean(dist_list))

        rouge_list = [rouge.compute(predictions=[gpt_eval], references=[label])['rougeL']
                      for gpt_eval, label in zip(gpt_eval_list, label_list)]
        dist_list = [1.0 - Levenshtein.distance(gpt_eval, label) / max(len(gpt_eval), len(label))
                     for gpt_eval, label in zip(gpt_eval_list, label_list)]
        print(f"GPT evaluation rougeL for {self.inst_name}: ", np.mean(rouge_list))
        print(f"GPT evaluation edit distance for {self.inst_name}: ", np.mean(dist_list))

        rouge_list = [rouge.compute(predictions=[human_eval], references=[label])['rougeL']
                      for human_eval, label in zip(human_eval_list, label_list)]
        dist_list = [1.0 - Levenshtein.distance(human_eval, label) / max(len(human_eval), len(label))
                     for human_eval, label in zip(human_eval_list, label_list)]
        print(f"Human evaluation rougeL for {self.inst_name}: ", np.mean(rouge_list))
        print(f"Human evaluation edit distance for {self.inst_name}: ", np.mean(dist_list))


class IFormatColor(EvalUnit):
    inst_name = 'i_format_color'

    def evaluate(self):
        pass

    def calculate_metrics(self):
        pass


if __name__ == '__main__':
    a = IFormatColor(model_name='Dalle3')
    a.evaluate()
    a.calculate_metrics()
