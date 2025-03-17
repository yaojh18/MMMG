import random
import os
import re
import shutil
import requests
from abc import abstractmethod
from transformers import MusicgenForConditionalGeneration

from prompt import *
from utils import *


class Model:
    model_name: str

    @abstractmethod
    def generate(self, query_list):
        """
        :param query_list: List[str]
        Format of an item in the response list:
        {
            "query": "Can you give me a step-by-step tutorial of how to make tomato soup?",
            "response": "Sure! Here is a step-by-step tutorial of how to make tomato soup: First, wash the tomatoes with clean water. <image_begin><image_1><image_end>. Second, ..."
            "image_list": [PIL.image, ...],
            "audio_list": [np.ndarray, ...]
        }
        """
        pass


# --------------------------- Below are tool models, can not be directly used for deployment -------------------------- #

class VoxInstruct(Model):
    def __init__(self):
        self.mllm_prompt = ('###Instrution:\n Your task is to generate a speech transcript based on a user\'s prompt. '
                            'The prompt is either generating a new transcript or modifying the original speech transcript (given in speech audio) to meet the format requirement. '
                            'You should output ONLY the final generated or modified transcript, omitting your thinking process.'
                            'Make sure you strictly follow the user\'s prompt. Your final answer should be always within 50 words whatever the user\'s prompt is.\n'
                            '### User\'s prompt:\n')
        self.mllm = 'gemini-1.5-pro'

    def generate(self, query_list, language='english'):
        """
        This model does not require a formated output list, thus can only be used for intermediate results.
        """
        # transcript_list = None
        # if not any(['audio_list' in query for query in query_list]):
        #     input_list = [f"{idx}|{0 if language == 'english' else 1}|{query[21:]}|\n" for idx, query in enumerate(query_list)]
        # else:
        #     if query_list[0]['instruction'].startswith('Read'):
        #         input_list = []
        #         for idx, query in enumerate(query_list):
        #             text = re.search(r'"(.*)"', query['instruction']).group(1)
        #             shutil.copy(query['audio_list'][0], f'./models/VoxInstruct/input/{idx}.wav')
        #             input_list.append(f'{idx}|0|\"{query["text_list"][0]} {text}\"|./input/{idx}.wav\n')
        #     else:
        #         audio_list = [[librosa.load(query['audio_list'][0])[0]] if query['audio_list'] else [] for query in query_list]
        #         mllm_query_list = [form_gemini_mm_query(self.mllm_prompt + query['instruction'], audios=audio)
        #                            for query, audio in zip(query_list, audio_list)]
        #         transcript_list = batch(query_gemini, data_list=mllm_query_list, model=self.mllm, temperature=0.0)
        #         input_list = []
        #         for idx, (query, transcript) in enumerate(zip(query_list, transcript_list)):
        #             input_list.append(f'{idx}|0|Read \"{transcript.strip()}\" in a common voice.|\n')
        input_list = []
        for idx, query in enumerate(query_list):
            if query['reference'] != '':
                shutil.copy(query['reference'], f'./models/VoxInstruct/input/{idx}.wav')
                input_list.append(f"{idx}|{int(language != 'english')}|\"{query['text']}\"|./input/{idx}.wav\n")
            else:
                input_list.append(f"{idx}|{int(language != 'english')}|{query['style']},\"{query['text']}\"|\n")
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

# --------------------------------------------- Tool Models end ------------------------------------------------------ #


class Dalle3(Model):
    def __init__(self, model_name='dall-e-3', revise=True):
        self.model_name = model_name
        self.revise = revise

    @staticmethod
    def generate_image_from_openai(index, prompt, model):
        client = openai.OpenAI(api_key=OPENAI_KEY)
        retry_count = 2
        retry_interval = 1

        for _ in range(retry_count):
            try:
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                )
                img_url = response.data[0].url
                img_res = requests.get(img_url)
                if img_res.status_code == 200:
                    image = BytesIO(img_res.content)
                    image = Image.open(image)
                    return index, image
                else:
                    raise ConnectionError

            except Exception as e:
                print("Error info: ", e)
                print('Retrying....')
                retry_interval *= 2
                time.sleep(retry_interval)
        print('Fail to get response.')
        return index, Image.new("RGB", (1024, 1024), "white")

    def generate(self, query_list):
        query_list = [('' if self.revise else 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:')
                      + query['instruction'] for query in query_list]
        image_list = batch(self.generate_image_from_openai, query_list, model=self.model_name)
        res_list = []
        for query, image in zip(query_list, image_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0),
                'image_list': [image],
                'audio_list': []
            })
        return res_list


class OmniGen(Model):
    def __init__(self):
        super().__init__()
        from OmniGen import OmniGenPipeline
        self.batch_size = 16
        self.sample_size = 4
        self.pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

    def generate(self, query_list):
        ### TODO: only image editing is implemented here
        text_list = ['<img><|image_1|></img>' + query['instruction'] for query in query_list]
        image_list = [query['image_list'] for query in query_list]
        output_list = []
        for begin in range(0, len(query_list), self.batch_size):
            end = begin + self.batch_size if begin + self.batch_size < len(query_list) else len(query_list)
            output_list += self.pipe(
                prompt=text_list[begin: end],
                input_images=image_list[begin: end],
                height=512,
                width=512,
                seed=0,
            )

        res_list = []
        for query, output in zip(query_list, output_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0),
                'image_list': [output],
                'audio_list': []
            })
        return res_list


class TangoFlux(Model):
    def __init__(self):
        super().__init__()
        from tangoflux import TangoFluxInference
        self.model = TangoFluxInference(name='declare-lab/TangoFlux')

    def generate(self, query_list):
        res_list = []
        random.seed(0)
        for query in query_list:
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [librosa.to_mono(self.model.generate(
                    query['instruction'], steps=50, duration=5, seed=random.randint(0, 1000)).numpy())],
            })
        return res_list


class MusicGen(Model):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large").to('cuda')

    def generate(self, query_list):
        query_list = [query['instruction'] for query in query_list]
        output_list = []
        for query in tqdm(query_list):
            inputs = self.processor(text=[query], padding=True, return_tensors="pt").to('cuda')
            output_list.append(librosa.resample(
                self.model.generate(**inputs, max_new_tokens=500)[0, 0].to('cpu').numpy(),
                orig_sr=self.model.config.audio_encoder.sampling_rate,
                target_sr=SAMPLE_RATE,
            ))
        res_list = []
        for query, output in zip(query_list, output_list):
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [output],
            })
        return res_list


class AudioAgent(Model):
    def __init__(self, mllm='gemini-1.5-pro'):
        self.mllm = GeminiModel(mllm, system_prompt=A_AGENT_PROMPT)
        self.models = (TangoFlux(), VoxInstruct(), MusicGen())

    def generate(self, query_list):
        responses = self.mllm.generate(query_list)
        output_list = []
        pattern = r'<audio_start>(.*?)</?audio_end>'
        audio_pattern = r'<[\s/]*audio_type="(sound|speech|music)"[\s/]*><[\s/]*audio_text="(.*?)"[\s/]*><[\s/]*audio_style=(?:"(.*?)"|(\d+)|(#\d+))[\s/]*>'
        for query, res in zip(query_list, responses):
            audio_prompts = re.findall(pattern, res)
            audio_list = []
            for i in range(len(audio_prompts)):
                audio_prompt = re.match(audio_pattern, audio_prompts[i])
                if audio_prompt is None:
                    audio_list.append(FAILED_TOKEN)
                    continue
                audio_prompt = audio_prompt.groups()
                if audio_prompt[2] is not None:
                    audio_list.append({
                        "type": audio_prompt[0], "text": audio_prompt[1],
                        "style": audio_prompt[2], "reference": ""
                    })
                elif audio_prompt[3] is not None:
                    audio_list.append({
                        "type": audio_prompt[0], "text": audio_prompt[1],
                        "style": "", "reference": query['audio_list'][int(audio_prompt[3])]
                    })
                elif audio_prompt[4] is not None:
                    ref_idx = int(audio_prompt[4][1:])
                    if ref_idx < len(audio_list) and (os.path.exists(audio_list[ref_idx]["reference"]) or audio_list[ref_idx]["style"]):
                        audio_list.append({
                            "type": audio_prompt[0], "text": audio_list[ref_idx]['text'] + ' ' + audio_prompt[1],
                            "style": "", "reference": ref_idx
                        })
                    else:
                        audio_list.append(FAILED_TOKEN)
                        continue
                else:
                    audio_list.append(FAILED_TOKEN)
                    continue
                old_tag = f"<audio_start>{audio_prompts[i]}<audio_end>"
                new_tag = f"<audio_start><audio_{i}><audio_end>"
                res = res.replace(old_tag, new_tag)
            output_list.append({
                'query': query,
                'response': res,
                'image_list': [],
                'audio_list': audio_list,
            })
        sound_query_list = [[], [], []]
        idx = [0, 0, 0]
        for output in output_list:
            for i in range(len(output['audio_list'])):
                if output['audio_list'][i] != FAILED_TOKEN and not isinstance(output['audio_list'][i]['reference'], int):
                    if output['audio_list'][i]['type'] == "sound":
                        sound_query_list[0].append({'instruction': output['audio_list'][i]['style']})
                        output['audio_list'][i] = 0, idx[0]
                        idx[0] += 1
                    elif output['audio_list'][i]['type'] == "speech":
                        sound_query_list[1].append(output['audio_list'][i])
                        output['audio_list'][i] = 1, idx[1]
                        idx[1] += 1
                    else:
                        sound_query_list[2].append({'instruction': output['audio_list'][i]['style']})
                        output['audio_list'][i] = 2, idx[2]
                        idx[2] += 1
        responses = [model.generate(query_list) for model, query_list in zip(self.models, sound_query_list)]
        responses = [[r['audio_list'][0] for r in res] for res in responses]
        tts_query_list = []
        idx = 0
        if os.path.exists('./output/AudioAgent/temp/'):
            shutil.rmtree('./output/AudioAgent/temp/')
        os.makedirs('./output/AudioAgent/temp/', exist_ok=True)
        for output in output_list:
            audio_map = {}
            for i in range(len(output['audio_list'])):
                if isinstance(output['audio_list'][i], tuple):
                    audio_map[i] = output['audio_list'][i]
                    output['audio_list'][i] = responses[audio_map[i][0]][audio_map[i][1]]
                    sf.write(f"./output/AudioAgent/temp/{audio_map[i][0]}_{audio_map[i][1]}.wav", output['audio_list'][i], SAMPLE_RATE)
                elif isinstance(output['audio_list'][i], dict):
                    ref_id = output['audio_list'][i]['reference']
                    output['audio_list'][i]['reference'] = f"./output/AudioAgent/temp/{audio_map[ref_id][0]}_{audio_map[ref_id][1]}.wav"
                    tts_query_list.append(output['audio_list'][i])
                    output['audio_list'][i] = idx
                    idx += 1
        responses = self.models[1].generate(tts_query_list)
        responses = [res['audio_list'][0] for res in responses]
        for output in output_list:
            for i in range(len(output['audio_list'])):
                if isinstance(output['audio_list'][i], int):
                    output['audio_list'][i] = responses[output['audio_list'][i]]
            output['audio_list'] = [a for a in output['audio_list'] if a is not FAILED_TOKEN]
        return output_list


class ImageAgent(Model):
    def __init__(self, mllm='gpt-4o-2024-11-20', diffusion='dalle3'):
        self.mllm = OpenAIModel(mllm, system_prompt=I_AGENT_PROMPT)
        self.diffusion = {
            'dalle3': Dalle3(revise=False)
        }[diffusion]

    def generate(self, query_list):
        responses = self.mllm.generate(query_list)
        diffusion_query_list = []
        output_list = []
        idx = 0
        pattern = r'<image_start>(.*?)</?image_end>'
        for query, res in zip(query_list, responses):
            image_prompts = re.findall(pattern, res)
            for i in range(len(image_prompts)):
                diffusion_query_list.append({'instruction': image_prompts[i]})
                old_tag = f"<image_start>{image_prompts[i]}<image_end>"
                new_tag = f"<image_start><image_{i}><image_end>"
                res = res.replace(old_tag, new_tag)
            output_list.append({
                'query': query,
                'response': res,
                'image_list': list(range(idx, idx + len(image_prompts))),
                'audio_list': [],
            })
            idx += len(image_prompts)
        res_list = self.diffusion.generate(diffusion_query_list)
        for output in output_list:
            output['image_list'] = [res_list[i]['image_list'][0] for i in output['image_list']]
        return output_list


class Anole(Model):
    def generate(self, query_list):
        os.makedirs('./models/Anole/input/', exist_ok=True)
        with open('./models/Anole/input/prompt.txt', 'w', encoding='utf-8') as f:
            f.writelines([query['instruction'] + '\n' for query in query_list])
        os.chdir("./models/Anole")
        if os.path.exists('./output'):
            shutil.rmtree('./output')
            print('History output has been removed!')
        os.system(f"python interleaved_generation.py")
        os.chdir("../..")
        output_list = []
        for idx, query in enumerate(query_list):
            dir_path = f'./models/Anole/output/{idx}/'
            with open(dir_path + 'response.txt', 'r', encoding='utf-8') as f:
                text = ''.join(f.readlines())
            image_list = [Image.open(dir_path + f) for f in os.listdir(dir_path) if f.endswith(".png")]
            output_list.append({
                'query': query,
                'response': text,
                'image_list': image_list,
                'audio_list': [],
            })
        return output_list
