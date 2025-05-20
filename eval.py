from model_image import *
from model_audio import *
from model_interleaved import *
from model_customized import *
from interface import *


class EvalUnit:
    inst_name: str

    def __init__(self, model_name: str, inst_name=None, sample_size=4):
        self.inst_list = []
        self.model_name = model_name
        self.sample_size = sample_size
        if inst_name is not None:
            self.inst_name = inst_name
        with open(f'./seed_instruction/{self.inst_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                self.inst_list.append(json.loads(line.strip()))
        self.inst_list = [inst.copy() for inst in self.inst_list for _ in range(self.sample_size)]

        if os.path.exists(f'./output/{model_name}/{self.inst_name}.jsonl'):
            self.load()
            if len(self.inst_list) == len(self.res_list):
                return
        if model_name.startswith('RandomModel_'):
            model = eval(f'{model_name.split("_")[0]}(sample_size={self.sample_size})')
            self.res_list = model.generate(self.inst_name)
        else:
            model = eval(f'{model_name}()')
            query_list = []
            for inst in self.inst_list:
                query = {'instruction': inst['instruction_para']}
                if 'image_list' in inst:
                    query['image_list'] = [f'./seed_instruction/image/{self.inst_name}_{idx}.png' for idx in inst['image_list']]
                else:
                    query['image_list'] = []
                if 'audio_list' in inst:
                    query['audio_list'] = [f'./seed_instruction/audio/{self.inst_name}_{idx}.wav' for idx in inst['audio_list']]
                else:
                    query['audio_list'] = []
                query_list.append(query)
            self.res_list = model.generate(query_list)
        self.save(save_all=True)
        self.load()

    def load(self):
        self.res_list = []
        with open(f'./output/{self.model_name}/{self.inst_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                self.res_list.append(json.loads(line.strip()))
        for res in self.res_list:
            image_list = []
            for image_name in res['image_list']:
                image_list.append(Image.open(f'./output/{self.model_name}/image/{self.inst_name}_{image_name}.png'))
            res['image_list'] = image_list
            audio_list = []
            for audio_name in res['audio_list']:
                audio, sr = librosa.load(f'./output/{self.model_name}/audio/{self.inst_name}_{audio_name}.wav')
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                audio_list.append(audio)
            res['audio_list'] = audio_list

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
                sf.write(output_path + f'audio/{self.inst_name}_{idx}.wav', audio, SAMPLE_RATE)

    def load_inst_mm(self):
        for inst in self.inst_list:
            if 'image_list' in inst:
                inst['image_list'] = [Image.open(f'./seed_instruction/image/{self.inst_name}_{idx}.png') for idx in inst['image_list']]
            if 'ref_image_list' in inst:
                inst['ref_image_list'] = [Image.open(f'./seed_instruction/image/{self.inst_name}_{idx}.png') for idx in inst['ref_image_list']]
            if 'audio_list' in inst:
                inst['audio_list'] = [librosa.load(f'./seed_instruction/audio/{self.inst_name}_{idx}.wav')[0] for idx in inst['audio_list']]

    def pad_inst_list(self):
        self.inst_list = [inst for inst in self.inst_list for _ in range(self.sample_size)]

    def evaluate(self):
        pass

    def human_evaluate(self):
        pass

    def compute_accuracy(self, return_list=False):
        pass

    def compute_correlation(self):
        return -1.0, -1.0
