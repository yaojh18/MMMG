import numpy as np
from sklearn.metrics import cohen_kappa_score

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


class IObjectInclude(EvalUnit):
    inst_name = 'i_object_include'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self):
        queries = []
        instruction = 'Is there {} in the given image? Answer only yes or no.'
        eval_inst_list = [instruction.format(inst['object']) for inst in self.inst_list for _ in
                          range(self.sample_size)]

        if not all(['gpt_eval' in res for res in self.res_list]):
            for res, inst in zip(self.res_list, eval_inst_list):
                queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + inst, images=res['image_list']))
            responses = batch_query_openai(queries, model_name='gpt-4o')
            responses = parse_responses(responses, pattern='(yes|no)',
                                        post_process=lambda x: 1.0 if x.lower() == 'yes' else 0.0)
            for data, response in zip(self.res_list, responses):
                data['gpt_eval'] = response
            self.save()

        if not all(['human_eval' in res for res in self.res_list]):
            interface = LabelInterface(eval_inst_list=eval_inst_list, data_list=self.res_list)
            interface.start()
            for data, human_eval in zip(self.res_list, interface.eval_list):
                data['human_eval'] = human_eval
            self.save()

        gpt_eval_list = [res['gpt_eval'] for res in self.res_list]
        human_eval_list = [res['human_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy for {self.inst_name}: ", np.mean(gpt_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(
            f"Cohen's Kappa for {self.inst_name}: ",
            1.0 if gpt_eval_list == human_eval_list else cohen_kappa_score(gpt_eval_list, human_eval_list, labels=[0.0, 1.0])
        )
        print(
            f"Pearson Correlation for {self.inst_name}: ",
            1.0 if gpt_eval_list == human_eval_list else np.corrcoef(gpt_eval_list, human_eval_list)[0, 1]
        )


if __name__ == '__main__':
    a = IObjectInclude(
        model_name='Dalle3'
    )
    a.evaluate()
