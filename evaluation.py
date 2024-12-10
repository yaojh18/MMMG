from sklearn.metrics import cohen_kappa_score
from typing import Callable

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


class IObject(EvalUnit):
    label_list: tuple

    @staticmethod
    @abstractmethod
    def instruction_func(inst: dict):
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
        queries = []
        eval_inst_list = [self.instruction_func(inst) for inst in self.inst_list for _ in range(self.sample_size)]

        if not all(['gpt_eval' in res for res in self.res_list]):
            for res, inst in zip(self.res_list, eval_inst_list):
                queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + inst, images=res['image_list']))
            responses = batch_query_openai(queries, model_name='gpt-4o')
            parsed_responses = [self.gpt_judge_process_func(res) for res in responses]
            for data, res in zip(self.res_list, parsed_responses):
                data['gpt_eval'] = res
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
            for data, inst in zip(self.res_list, [inst for inst in self.inst_list for _ in range(self.sample_size)]):
                data['human_eval'] = float(data['human_eval'] == (inst['count'] - 2))
                data['gpt_eval'] = float(data['gpt_eval'] == (inst['count'] - 2))
            self.save()

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
        return f"How many {inst['object']} are there in the given image? Choose from the options:\nA. Less than 3\nB. 3\nC. 4\nD. 5\nE. 6\nF. More than 6\n Respond only with the option letter (A, B, C, D, E or F). Do not provide any explanation, reasoning, or additional information."

    @staticmethod
    def gpt_judge_process_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd', 'e', 'f') else FAILED_TOKEN

    @staticmethod
    def human_judge_process_func(res: str):
        return res


if __name__ == '__main__':
    a = IObjectCounting(model_name='Dalle3')
    a.evaluate()
