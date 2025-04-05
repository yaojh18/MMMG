import os
import re
import shutil
from abc import abstractmethod

from utils import *


class Model:
    model_name: str

    @abstractmethod
    def generate(self, query_list):
        """
        :param query_list: List[dict]
        Format of an item in the response list:
        {
            "query": {"instruction": "Can you give me a step-by-step tutorial of how to make tomato soup?", "image_list": [], "audio_list": []},
            "response": "Sure! Here is a step-by-step tutorial of how to make tomato soup: First, wash the tomatoes with clean water. <image_begin><image_1><image_end>. Second, ..."
            "image_list": [PIL.image, ...],
            "audio_list": [np.ndarray, ...]
        }
        """
        pass

### Tool models

class VoxInstruct(Model):
    def generate(self, query_list):
        """
        This model does not require a formated output list, thus can only be used for intermediate results.
        """
        input_list = []
        language = 'chinese' if re.search(r'[\u4e00-\u9fff]', query_list[0]['text']) is not None else 'english'
        for idx, query in enumerate(query_list):
            if query['reference'] != '':
                shutil.copy(query['reference'], f'./models/VoxInstruct/input/{idx}.wav')
                input_list.append(f"{idx}|{int(language != 'english')}|\"{query['reference_text']} {query['text']}\"|./input/{idx}.wav\n")
            else:
                input_list.append(f"{idx}|{int(language != 'english')}|{query['style']}, \"{query['text']}\"|\n")
        with open('./models/VoxInstruct/input/instructions.txt', 'w', encoding='utf-8') as f:
            f.writelines(input_list)
        os.chdir("./models/VoxInstruct")
        if os.path.exists('./output'):
            shutil.rmtree('./output')
            os.makedirs('./output')
        os.system("./infer.sh")
        os.chdir("../..")
        res_list = []
        for idx, query in enumerate(query_list):
            audio, sr = librosa.load(f'./models/VoxInstruct/output/{idx}.wav')
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [audio],
            })
        return res_list


class OpenAIModel(Model):
    def __init__(self, model_name, system_prompt=''):
        self.model_name = model_name
        self.system_prompt = [{"role": "developer", "content": system_prompt}] if system_prompt else []

    def generate(self, query_list):
        """
        This model will not return a formated output list, thus can only be used for intermediate results.
        """
        mllm_query_list = [self.system_prompt + form_openai_mm_query(
            query['instruction'] + (IMAGE_TOKEN(0) if 'image_list' in query else ''),
            images=[Image.open(image) for image in query['image_list']] if 'image_list' in query else []
        ) for query in query_list]
        return batch(query_openai, mllm_query_list, model=self.model_name, temperature=0.2)


class GeminiModel(Model):
    def __init__(self, model_name, system_prompt=''):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def generate(self, query_list):
        """
        This model will not return a formated output list, thus can only be used for intermediate results.
        """
        mllm_query_list = [form_gemini_mm_query(
            '### System:\n' + self.system_prompt +'\n### User:\n' + query['instruction'],
            images=[Image.open(image) for image in query['image_list']] if 'image_list' in query else [],
            audios=[librosa.load(audio)[0] for audio in query['audio_list']] if 'audio_list' in query else [],
        ) for query in query_list]
        return batch(query_gemini, mllm_query_list, model=self.model_name, temperature=0.2)
