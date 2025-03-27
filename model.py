import random
import os
import re
import shutil
import sys
import subprocess
import requests
from abc import abstractmethod

from prompt import *
from utils import *
import PIL

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


class Imagen(Model): # should we also add para "revise" here as Dalle3 does?
    def __init__(self, model_name='imagen-3.0-generate-002'):
        self.model_name = model_name
        self.client = genai.Client(api_key=GEMINI_KEY)

    @staticmethod
    def generate_image_from_google(self, index, prompt, num_images=1):
        retry_count = 0
        max_retries = 3
        retry_interval = 1
        while retry_count < max_retries:
            try:
                response = self.client.models.generate_images(
                    model=self.model_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=num_images,
                        aspect_ratio='1:1',  # or '16:9', '4:3',etc
                    )
                )

                images = [
                    Image.open(BytesIO(generated_image.image.image_bytes))
                    for generated_image in response.generated_images
                ]
                return index, images

            except Exception as e:
                print(f"Error info: {e}")
                print('Retrying....')
                retry_count += 1
                time.sleep(retry_interval * (2 ** retry_count)) 

        print('Failed to get response.')
        return index, [None] * num_images  

    def generate(self, query_list):
        prompts = [query['instruction'] for query in query_list]

        results = []
        for index, prompt in enumerate(prompts):
            _, images = self.generate_image_from_google(index, prompt, num_images=1) 
            results.append({
                'query': query_list[index],
                'response': IMAGE_TOKEN(0),
                'image_list': images, 
                'audio_list': []
            })

        return results
    
class StableDiffusion3_5(Model):
    def __init__(self, model_name="stabilityai/stable-diffusion-3.5-large"):
        from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
    
        self.model_name = model_name
        self.device = torch.device("cuda:0")
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16,
            token=HF_KEY,
            trust_remote_code=True
        )

        transformer_model = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            # quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            token=HF_KEY,
            trust_remote_code=True
        )

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_name,
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
            token=HF_KEY,
            trust_remote_code=True
        )

        self.pipeline.enable_model_cpu_offload()

    @staticmethod
    def _process_image(image):
        return image if isinstance(image, Image.Image) else Image.fromarray(image)

    def generate_image(self, index, prompt, num_inference_steps=28, guidance_scale=4.5, max_sequence_length=512):
        try:
            output = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
            )
            image = self._process_image(output.images[0])
            return index, image

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            return index, None 

    def generate(self, query_list):
        res_list = []
        for index, query in enumerate(query_list):
            prompt = query.get("instruction", "")
            _, image = self.generate_image(
                index=index,
                prompt=prompt,
                num_inference_steps=query.get("num_inference_steps", 28),
                guidance_scale=query.get("guidance_scale", 4.5),
                max_sequence_length=query.get("max_sequence_length", 512),
            )

            res_list.append({
                "query": query,
                "response": IMAGE_TOKEN(0),
                "image_list": [image] if image else [],
                "audio_list": [],
            })

        return res_list


class Recraft(Model):
    def __init__(self, model_name="recraft-ai/recraft-v3"):
        self.model_name = model_name
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_KEY

    def generate(self, query_list):
        import replicate
        res_list = []
        for query in query_list:
            try:
                prompt = query.get("instruction", "")
                size = query.get("size", "1024x1024")

                input_params = {
                    "size": size,
                    "prompt": prompt
                }

                output = replicate.run(
                    self.model_name,
                    input=input_params,
                )

                image_bytes = output.read()
                image = Image.open(BytesIO(image_bytes))

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [image],
                    "audio_list": [],
                })
                

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [],
                    "audio_list": [],
                })

        return res_list    


class LumaPhoton(Model):
    def __init__(self, model_name="luma/photon"):
        self.model_name = model_name
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_KEY

    def generate(self, query_list):
        import replicate
        res_list = []
        for query in query_list:
            try:
                prompt = query.get("instruction", "")
                size = query.get("size", "1024x1024")

                input_params = {
                    "size": size,
                    "prompt": prompt
                }

                output = replicate.run(
                    self.model_name,
                    input=input_params,
                )

                image_bytes = output.read()
                image = Image.open(BytesIO(image_bytes))

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [image],
                    "audio_list": [],
                })
                

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [],
                    "audio_list": [],
                })

        return res_list    

class Flux1_1(Model):
    def __init__(self, model_name="black-forest-labs/flux-1.1-pro"):
        self.model_name = model_name
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_KEY

    def generate(self, query_list):
        import replicate
        res_list = []
        for query in query_list:
            try:
                prompt = query.get("instruction", "")
                size = query.get("size", "768x1344") # WARNING: If the size is too large (e.g. 1024*1024), there may be error: Prediction interrupted and the image should be uploaded via URLs (github issue#135) 

                input_params = {
                    "prompt": prompt,
                    "prompt_upsampling": True,
                    "size":size
                }

                output = replicate.run(
                    self.model_name,
                    input=input_params,
                )

                image_bytes = output.read()
                image = Image.open(BytesIO(image_bytes))

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [image],
                    "audio_list": [],
                })
                

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [],
                    "audio_list": [],
                })

        return res_list    

class Ideogram_v2(Model):
    def __init__(self, model_name="ideogram-ai/ideogram-v2"):
        self.model_name = model_name
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_KEY

    def generate(self, query_list):
        import replicate
        res_list = []
        for query in query_list:
            try:
                prompt = query.get("instruction", "")

                input_params = {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",  # or "16:9", "4:3", etc.
                }

                output = replicate.run(
                    self.model_name,
                    input=input_params,
                )

                image_bytes = output.read()
                image = Image.open(BytesIO(image_bytes))

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [image],
                    "audio_list": [],
                })
                

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [],
                    "audio_list": [],
                })

        return res_list  
    

class Flux_1_dev(Model):
    def __init__(self, model_name="FLUX.1-dev"):
        super().__init__()
        self.model_name = model_name

        from diffusers import FluxPipeline        
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            token=HF_KEY,
            trust_remote_code=True
        )
        self.pipe.enable_model_cpu_offload()  
        
        self.default_params = {
            'height': 1024,
            'width': 1024,
            'guidance_scale': 3.5,
            'num_inference_steps': 50,
            'max_sequence_length': 512,
            'generator': torch.Generator("cpu").manual_seed(0)
        }

    def generate(self, query_list):
        res_list = []
        for query in query_list:
            prompt = query['instruction']
            params = self.default_params.copy()

            if 'params' in query:
                params.update(query['params'])
            
            image = self.pipe(
                prompt,
                height=params['height'],
                width=params['width'],
                guidance_scale=params['guidance_scale'],
                num_inference_steps=params['num_inference_steps'],
                max_sequence_length=params['max_sequence_length'],
                generator=params['generator']
            ).images[0]
            
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
        from transformers import MusicgenForConditionalGeneration
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

class Emu3(Model):
    def __init__(self):
        super().__init__()
        
        from models.emu3.mllm.processing_emu3 import Emu3Processor # extra path 
        EMU_HUB = "BAAI/Emu3-Gen"
        VQ_HUB = "BAAI/Emu3-VisionTokenizer"

        self.model = AutoModelForCausalLM.from_pretrained(
            EMU_HUB,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
        self.image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(VQ_HUB, trust_remote_code=True).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        self.generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

    def generate(self, query_list):
        res_list = []
        for query in tqdm(query_list):
            text = query['instruction']
            images = query.get('image_list', [])
            inputs = self.processor(
                text=text,
                images=images,
                mode='G',
                ratio="1:1",
                image_area=self.model.config.image_area,
                return_tensors="pt",
                padding="longest",
            )

            h = inputs.image_size[:, 0]
            w = inputs.image_size[:, 1]
            constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
            logits_processor = LogitsProcessorList([
                PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
            ])

            outputs = self.model.generate(
                inputs.input_ids.to("cuda:0"),
                self.generation_config,
                logits_processor=logits_processor,
                attention_mask=inputs.attention_mask.to("cuda:0"),
            )

            decoded_outputs = self.processor.decode(outputs[0])
            image_list = [im for im in decoded_outputs if isinstance(im, Image.Image)]

            res_list.append({
                'query': query,
                'response': ''.join([str(item) for item in decoded_outputs if not isinstance(item, Image.Image)]),
                'image_list': image_list,
                'audio_list': [],
            })

        return res_list
      
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


class Janus(Model):
    def __init__(self):
        super().__init__()
        from models.janus.models import MultiModalityCausalLM, VLChatProcessor
        from models.janus.utils.io import load_pil_images

        self.model_path = "deepseek-ai/Janus-Pro-7B"
        
        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(torch.bfloat16).cuda().eval()

    def generate(self, query_list):
        res_list = []
        for query in tqdm(query_list):
            conversation = [
                {"role": "<|User|>", "content": query['instruction']},
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag

            generated_images = self._generate_images(prompt)

            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0),
                'image_list': generated_images,
                'audio_list': [],
            })

        return res_list

    @torch.inference_mode()
    def _generate_images(self, prompt, temperature=1.0, parallel_size=16, cfg_weight=5.0, img_size=384, patch_size=16):
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        image_token_num_per_image = 576
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        past_key_values = None
        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            past_key_values = outputs.past_key_values

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        os.makedirs('./output/janus/generated_images', exist_ok=True) # how should I name the path?
        image_list = []
        for i in range(parallel_size):
            save_path = os.path.join('./output/janus/generated_images', f"img_{i}.jpg")
            img = PIL.Image.fromarray(dec[i])
            img.save(save_path)
            image_list.append(img)

        return image_list
    
# FIXME: path issues for vila-u
class VilaU(Model):
    def __init__(self, model_path="./models/vila-u/vila-uvila-u-7b-256", vila_u_path="./models/vila-u"):
        super().__init__()
        self._add_vila_u_path(vila_u_path)
        self.model = self._load_model(model_path)
        self.save_path = "./output/vila-u/generated_images/" # how should I name the path?
        os.makedirs(self.save_path, exist_ok=True)

    def _add_vila_u_path(self, vila_u_path): # to import vila-u from the right path
        abs_path = os.path.abspath(vila_u_path)
        if abs_path not in sys.path:
            sys.path.append(abs_path)
            
    def _load_model(self, model_path):
        try:
            import models.vilau.vila_u
            return models.vilau.vila_u.load(model_path)
        except ImportError:
            raise ImportError("The vila_u module is required to run this model.")

    def _save_image(self, response, path):
        """Save generated images to disk."""
        import cv2
        os.makedirs(path, exist_ok=True)
        image_list = []
        for i in range(response.shape[0]):
            image = response[i].permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            image = image.cpu().numpy().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(path, f"image_{i}.png")
            cv2.imwrite(save_path, image)
            image_list.append(Image.open(save_path))
        return image_list

    def generate(self, query_list):
        res_list = []
        for query in query_list:
            if "prompt" in query:
                prompt = query["prompt"]
                cfg = query.get("cfg", 3.0)
                generation_nums = query.get("generation_nums", 1)

                # image only by default, no video task
                response = self.model.generate_image_content(prompt, cfg, generation_nums)
                media_list = self._save_image(response, self.save_path)

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": media_list,
                    "audio_list": [],
                })
            elif "query" in query:
                text_query = query["query"]
                image_path = query.get("image_path")

                if image_path:
                    image = self._load_image(image_path)
                    response = self.model.generate_content([image, text_query])
                else:
                    raise ValueError("No visual content input!")

                res_list.append({
                    "query": query,
                    "response": response,
                    "image_list": [],
                    "audio_list": [],
                })
            else:
                raise ValueError("Invalid query format!")

        return res_list

    def _load_image(self, image_path):
        """Load an image using the vila_u utility."""
        try:
            import models.vilau.vila_u
            return models.vilau.vila_u.Image(image_path)
        except ImportError:
            raise ImportError("The vila_u module is required to load images.")

class LaVIT(Model): # FIXME: haven't resolved the env issues
    def __init__(self):
        super().__init__()
        abs_path = os.path.abspath("./models/LaVIT")
        if abs_path not in sys.path:
            sys.path.append(abs_path)
        
        self.model_path = "./models/LaVIT/LaVIT-7B-v2"
        self.model_dtype = "bf16"
        self.device_id = 0
        self.device = torch.device(f"cuda:{self.device_id}")
        self.torch_dtype = torch.bfloat16 if self.model_dtype == "bf16" else torch.float16

        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = self._build_model()

        self.ratio_dict = {
            "1:1": (1024, 1024),
            "4:3": (896, 1152),
            "3:2": (832, 1216),
            "16:9": (768, 1344),
            "2:3": (1216, 832),
            "3:4": (1152, 896),
        }

    def _build_model(self):
        try:
            from .models.LaVIT.models import build_model 
            model = build_model(
                model_path=self.model_path,
                model_dtype=self.model_dtype,
                check_safety=False,
                device_id=self.device_id,
                use_xformers=True,
                understanding=False,
            )
            return model.to(self.device)
        except ImportError:
            raise ImportError("The LaVIT module could not be loaded. Ensure the path is correct.")

    def _get_image_size(self, ratio="1:1"):
        if ratio not in self.ratio_dict:
            raise ValueError(f"Unsupported aspect ratio: {ratio}. Supported ratios are {list(self.ratio_dict.keys())}.")
        return self.ratio_dict[ratio]

    def generate(self, query_list):
        res_list = []
        for query in query_list:
            if "prompt" in query:
                # text-to-image
                prompt = query["prompt"]
                ratio = query.get("ratio", "1:1")
                guidance_scale_for_llm = query.get("guidance_scale", 4.0)
                num_return_images = query.get("num_return_images", 1)

                height, width = self._get_image_size(ratio)
                with torch.cuda.amp.autocast(enabled=True, dtype=self.torch_dtype):
                    images = self.model.generate_image(
                        prompt=prompt,
                        width=width,
                        height=height,
                        guidance_scale_for_llm=guidance_scale_for_llm,
                        num_return_images=num_return_images,
                    )
                image_list = [Image.fromarray(np.array(img)) for img in images]

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": image_list,
                    "audio_list": [],
                })

            elif "input_prompts" in query:
                input_prompts = query["input_prompts"]
                ratio = query.get("ratio", "1:1")
                guidance_scale_for_llm = query.get("guidance_scale", 5.0)
                num_return_images = query.get("num_return_images", 1)

                height, width = self._get_image_size(ratio)
                with torch.cuda.amp.autocast(enabled=True, dtype=self.torch_dtype):
                    images = self.model.multimodal_synthesis(
                        input_prompts=input_prompts,
                        width=width,
                        height=height,
                        guidance_scale_for_llm=guidance_scale_for_llm,
                        num_return_images=num_return_images,
                    )
                image_list = [Image.fromarray(np.array(img)) for img in images]

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": image_list,
                    "audio_list": [],
                })
            else:
                raise ValueError("Invalid query format!")

        return res_list
    
class QwenAudio(Model): # Qwen2-Audio-7B 
    def __init__(self):
        super().__init__()
        from transformers import Qwen2AudioForConditionalGeneration
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B", trust_remote_code=True
        )
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        
    def generate(self, query_list):
        res_list = []
        for query in query_list:
            instruction = query['instruction']            
            audio_signal = None
            if 'audio_url' in query:
                url = query['audio_url']
                audio_signal, _ = librosa.load(
                    BytesIO(urlopen(url).read()), sr=self.sample_rate
                )
            elif 'audio_list' in query and query['audio_list']:
                audio_signal, _ = librosa.load(query['audio_list'][0], sr=self.sample_rate)

            prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{instruction}"
            
            inputs = self.processor(
                text=prompt,
                audios=audio_signal if audio_signal is not None else None,
                return_tensors="pt"
            )
            generated_ids = self.model.generate(**inputs, max_length=256)
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            res_list.append({
                'query': instruction,
                'response': AUDIO_TOKEN(0) + response_text,
                'image_list': [],
                'audio_list': [audio_signal] if audio_signal is not None else [],
            })

        return res_list

# # TODO: import the module _image_to_audio from riffusiom/cli.py
# class Riffusion(Model):
#     def __init__(self):
#         from diffusers import DiffusionPipeline
#         self.model = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1").to('cuda')
#         self.sample_rate = 44100

#     def generate(self, query_list):
#         query_list = [query['instruction'][9:] for query in query_list]
#         output_list = []

#         for query in tqdm(query_list, desc="Generating audio"):
#             # spectrogram image
#             image = self.model(prompt=query).images[0]
#             audio_signal = self._image_to_audio(image)

#             audio_signal = librosa.resample(
#                 audio_signal,
#                 orig_sr=self.sample_rate,
#                 target_sr=SAMPLE_RATE
#             )
#             output_list.append(audio_signal)
        
#         res_list = []
#         for query, output in zip(query_list, output_list):
#             res_list.append({
#                 'query': query,
#                 'response': AUDIO_TOKEN(0),  
#                 'image_list': [],     
#                 'audio_list': [output],  
#             })

#         return res_list

#     def _image_to_audio(*, image: str, audio: str, device: str = "cuda"):
#         """
#         Reconstruct an audio clip from a spectrogram image.
#         """
#         pil_image = Image.open(image)

#         # Get parameters from image exif
#         img_exif = pil_image.getexif()
#         assert img_exif is not None

#         try:
#             params = SpectrogramParams.from_exif(exif=img_exif)
#         except (KeyError, AttributeError):
#             print("WARNING: Could not find spectrogram parameters in exif data. Using defaults.")
#             params = SpectrogramParams()

#         converter = SpectrogramImageConverter(params=params, device=device)
#         segment = converter.audio_from_spectrogram_image(pil_image)

#         extension = Path(audio).suffix[1:]
#         segment.export(audio, format=extension)

#         print(f"Wrote {audio} ({segment.duration_seconds:.2f} seconds)")



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
