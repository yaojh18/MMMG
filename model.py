import os
import shutil
import random
from abc import abstractmethod

from utils import *


class Model:
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

class BlankAudioModel(Model):
    def generate(self, query_list):
        output_list = []
        for query in query_list:
            output_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [np.zeros(SAMPLE_RATE)],
            })
        return output_list


class RandomModel(Model):
    def __init__(self, sample_size=2):
        assert sample_size >= 2
        self.sample_size = sample_size

    @staticmethod
    def inst_map(inst_name):
        if inst_name.startswith('i_consistency') or inst_name.startswith('i_structure') or inst_name.startswith('it'):
            return ['HybridAgent', 'GeminiAgent', 'Gemini2']
        if inst_name.startswith('i_edit'):
            return ['HybridAgent', 'Gemini2']
        if inst_name.startswith('i'):
            return ['Imagen3', 'Recraft3', 'LumaPhoton', 'Flux1_1Pro', 'Ideogram2', 'Dalle3']
        if inst_name.startswith('a_sound'):
            return ['StableAudio', 'AudioLDM2', 'AudioGen', 'Tango2', 'MakeAnAudio2']
        if inst_name.startswith('a_music'):
            return ['StableAudio', 'AudioLDM2', 'MusicGen', 'TangoMusic', 'YuE']
        if inst_name.startswith('a_speech') or inst_name.startswith('a_consistency') or inst_name.startswith('a_structure'):
            return ['VoxInstructAgent', 'VoiceLDMAgent']
        raise NotImplementedError(inst_name)

    def generate(self, inst_name):
        from eval import EvalUnit
        model_name_list = self.inst_map(inst_name)
        for model_name in model_name_list:
            if not os.path.exists(f'./output/{model_name}/{inst_name}.jsonl'):
                raise FileNotFoundError(f'./output/{model_name}/{inst_name}.jsonl')
        model_list = [EvalUnit(model_name=model_name, inst_name=inst_name, sample_size=4)
                      for model_name in model_name_list]
        output_list = []
        random.seed(0)

        for i in range(len(model_list[0].res_list) // model_list[0].sample_size):
            model_idxs = random.sample(range(len(model_list)), self.sample_size)
            for idx in model_idxs:
                gen_idx = random.randint(0, model_list[idx].sample_size - 1)
                output_list.append(model_list[idx].res_list[i * model_list[idx].sample_size + gen_idx])
        return output_list


class VoxInstruct(Model):
    def generate(self, query_list):
        """
        This model does not require a formated output list, thus can only be used for intermediate results.
        """
        audio_list = []
        for query in query_list:
            if 'reference' in query and query['reference'] != '':
                audio_list.append(librosa.load(query['reference'])[0])
        language = 'chinese' if re.search(r'[\u4e00-\u9fff]', query_list[0]['text']) is not None else 'english'
        transcripts, _ = transcribe_speech(audio_list, language=language)
        idx = 0
        for query in query_list:
            if 'reference' in query and query['reference'] != '':
                query['reference_text'] = transcripts[idx]
                idx += 1
        input_list = []
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
            audio, sr = librosa.load(f'./models/VoxInstruct/output/{idx}.wav', sr=SAMPLE_RATE)
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [audio],
            })
        return res_list


class VoiceLDM(Model):
    def __init__(self):
        super().__init__()
        from voiceldm import VoiceLDMPipeline
        self.model = VoiceLDMPipeline(device="cuda:0")
        self.num_inference_steps = 50
        self.desc_guidance_scale = 7
        self.cont_guidance_scale = 7
        
    def generate(self, query_list, language='english'):
        """
        This model does not require a formated output list, thus can only be used for intermediate results.
        """
        if os.path.exists('./models/VoiceLDM/input/'):
            shutil.rmtree('./models/VoiceLDM/input/')
            os.makedirs('./models/VoiceLDM/input/')
        else:
            os.makedirs('./models/VoiceLDM/input/')
            
        res_list = []
        for idx, query in enumerate(query_list):
            cont_prompt = query['text']
            if query['reference'] != '':                
                shutil.copy(query['reference'], f'./models/VoiceLDM/input/{idx}.wav')
                audio_prompt = f'./models/VoiceLDM/input/{idx}.wav'
                desc_prompt = None
            else:
                audio_prompt = None
                desc_prompt = query['style']
                    
            audio = self.model(
                desc_prompt=desc_prompt,
                cont_prompt=cont_prompt,
                audio_prompt=audio_prompt,
                num_inference_steps=self.num_inference_steps,
                desc_guidance_scale=self.desc_guidance_scale,
                cont_guidance_scale=self.cont_guidance_scale,
                device="cuda:0",
            )
        
            audio=audio[0].float().cpu().numpy()
            audio=librosa.resample(audio, orig_sr=16000, target_sr=SAMPLE_RATE)            
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
        mllm_query_list = [self.system_prompt + form_openai_mm_query(query['instruction'],
            images=[Image.open(image) for image in query['image_list']] if 'image_list' in query else []
        ) for query in query_list]
        return batch(query_openai, mllm_query_list, model=self.model_name, temperature=0.0)


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
        return batch(query_gemini, mllm_query_list, model=self.model_name, temperature=0.0)
