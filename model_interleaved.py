import sys

from model import *
from model_image import *
from model_audio import *
from utils import *
from prompt import I_AGENT_PROMPT, A_AGENT_PROMPT
from pathlib import Path


### Image generation and editing model
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


### Agent models
class AudioAgent(Model):
    sound_model_name = 'BlankAudioModel'
    speech_model_name = 'VoxInstruct'
    music_model_name = 'BlankAudioModel'

    def __init__(self, mllm='gemini-1.5-pro'):
        self.mllm = GeminiModel(mllm, system_prompt=A_AGENT_PROMPT)
        self.models = (eval(f'{self.sound_model_name}()'), eval(f'{self.speech_model_name}()'), eval(f'{self.music_model_name}()'))

    def generate(self, query_list):
        responses = self.mllm.generate(query_list)
        output_list = []
        pattern = r'<audio_start>(.*?)</?audio_end>'
        audio_pattern = r'<[\s/]*audio_type="(sound|speech|music)"[\s/]*><[\s/]*audio_text="(.*?)"[\s/]*><[\s/]*audio_style=(?:"(.*?)"|(\d+)|(#\d+))[\s/]*>'
        for query, res in zip(query_list, responses):
            audio_prompts = re.findall(pattern, res)
            res = res.replace("</audio_end>", "<audio_end>")
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
                    ref_idx = int(audio_prompt[3])
                    if ref_idx < len(query['audio_list']):
                        audio_list.append({
                            "type": audio_prompt[0], "text": audio_prompt[1],
                            "style": "", "reference": query['audio_list'][ref_idx],
                            "reference_text": query['text_list'][ref_idx]
                        })
                    else:
                        audio_list.append(FAILED_TOKEN)
                        continue
                elif audio_prompt[4] is not None:
                    ref_idx = int(audio_prompt[4][1:])
                    if ref_idx < len(audio_list) and not isinstance(audio_list[ref_idx]["reference"], int):
                        audio_list.append({
                            "type": audio_prompt[0], "text": audio_prompt[1],
                            "style": "", "reference": ref_idx, "reference_text": audio_list[ref_idx]['text']
                        })
                    else:
                        audio_list.append(FAILED_TOKEN)
                        continue
                else:
                    audio_list.append(FAILED_TOKEN)
                    continue
                res = res.replace(audio_prompts[i], f"<audio_{i}>")
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


class VoxInstructAgent(AudioAgent):
    speech_model_name = 'VoxInstruct'


class VoiceLDMAgent(AudioAgent):
    speech_model_name = 'VoiceLDM'


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
            res = res.replace("</image_end>", "<image_end>")
            for i in range(len(image_prompts)):
                diffusion_query_list.append({'instruction': image_prompts[i]})
                res = res.replace(image_prompts[i], f"<image_{i}>")
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


### Interleaved I+T model

class Gemini2Flash(Model):
    model_name = 'gemini-2.0-flash-exp'
    def __init__(self):
        super().__init__()

    def generate(self, query_list):
        client = genai.Client(api_key=GEMINI_KEY)
        res_list = []
        
        for query in tqdm(query_list):
            try:
                contents = []
                text = query.get("instruction", "")
                if text:
                    contents.append(text)
                
                images = query.get("image_list", [])
                for img_path in images:
                    contents.append(Image.open(img_path))
                
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"]
                    )
                )

                generated_text = ""
                generated_images = []
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        generated_text += part.text
                    if part.inline_data is not None:
                        generated_images.append(Image.open(BytesIO(part.inline_data.data)))

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0) + generated_text, # TODO: multiple image tokens handling
                    "image_list": generated_images,
                    "audio_list": [],
                })

            except Exception as e:
                print(f"Error processing query: {query}. Error: {e}")
                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0),
                    "image_list": [Image.new("RGB", (1024, 1024), "white")],
                    "audio_list": [],
                })

        return res_list


class Anole(Model):
    def __init__(self):
        super().__init__()
        from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

        self.model = ChameleonForConditionalGeneration.from_pretrained(
            "leloy/Anole-7b-v0.1-hf",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
            token=HF_KEY,
        )
        self.processor = ChameleonProcessor.from_pretrained(
            "leloy/Anole-7b-v0.1-hf",
            trust_remote_code=True,
            token=HF_KEY,
        )
    
    def generate(self, query_list):
        import subprocess
        
        output_list = []
        os.makedirs('./output/Anole/', exist_ok=True)

        for idx, query in enumerate(tqdm(query_list)):
            try:
                instruction = query.get("instruction", "")
                image_paths = query.get("image_list", [])
                
                path_prefix = './models/Anole/'
                if not image_paths:  # Text-only generation
                    script_path = path_prefix + "scripts/text_only_generation.py"
                    command = [
                        "python", script_path,
                        "--prompt", instruction
                    ]
                elif len(image_paths) == 1:  # Text-image to text or text-image to image generation
                    if query.get("inference_mode") == "text-image-to-text":
                        script_path = path_prefix + "scripts/text_only_generation.py"
                        command = [
                            "python", script_path,
                            "--inference_mode", "text-image-to-text",
                            "--prompt", instruction,
                            "--image_1_path", image_paths[0]
                        ]
                    else:  # text-image to image
                        script_path = path_prefix + "scripts/image_only_generation.py"
                        command = [
                            "python", script_path,
                            "--inference_mode", "text-image-to-image",
                            "--prompt", instruction,
                            "--image_1_path", image_paths[0],
                            "--max_new_tokens", "2500"
                        ]
                elif len(image_paths) == 2:  # Multi-image to text or multi-image to image generation
                    if query.get("inference_mode") == "multi-image-to-text":
                        script_path = path_prefix + "scripts/text_only_generation.py"
                        command = [
                            "python", script_path,
                            "--inference_mode", "multi-image-to-text",
                            "--prompt", instruction,
                            "--image_1_path", image_paths[0],
                            "--image_2_path", image_paths[1]
                        ]
                else:  # Interleaved text and image generation
                    script_path = path_prefix + "scripts/interleaved_generation.py"
                    command = [
                        "python", script_path,
                        "--inference_mode", "text-to-interleaved-text-image",
                        "--prompt", instruction,
                        "--max_new_tokens", "2055"
                    ]  
                    
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Script execution failed: {result.stderr}")

                output = result.stdout.strip()
                response = output.split("Response:")[1].strip() if "Response:" in output else ""
                image_list = [Image.open(path) for path in image_paths] if image_paths else []

                output_list.append({
                    "query": query,
                    "response": response,
                    "image_list": image_list,
                    "audio_list": [],
                })

            except Exception as e:
                print(f"Error generating content for query: {query}. Error: {e}")
                default_image = Image.new("RGB", (1024, 1024), "white")
                
                output_list.append({
                    "query": query,
                    "response": "<image_start><image_{0}><image_end>",
                    "image_list": [default_image],
                    "audio_list": [],
                })
                
        return output_list



class Emu3(Model):
    model_name = 'Emu3'
    def __init__(self):
        super().__init__()

        from models.emu3.mllm.processing_emu3 import Emu3Processor
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        EMU_HUB = "BAAI/Emu3-Gen"
        VQ_HUB = "BAAI/Emu3-VisionTokenizer"

        self.model = AutoModelForCausalLM.from_pretrained(
            EMU_HUB,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            token=HF_KEY,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left",token=HF_KEY)
        self.image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True,token=HF_KEY)
        self.image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True, token=HF_KEY).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

    def generate(self, query_list):
        from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor
        from transformers.generation.configuration_utils import GenerationConfig

        res_list = []
        for query in tqdm(query_list):
            try:
                instruction = query.get("instruction", "")
                images = query.get("image_list", [])

                inputs = self.processor(
                    text=instruction,
                    images=images,
                    mode="G",  # Image gen mode by default
                    ratio="1:1",
                    image_area=self.model.config.image_area,
                    return_tensors="pt",
                    padding="longest",
                )
                inputs.input_ids = inputs.input_ids.to("cuda:0")
                inputs.attention_mask = inputs.attention_mask.to("cuda:0")
                
                h = inputs.image_size[:, 0]
                w = inputs.image_size[:, 1]
                constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
                
                logits_processor = LogitsProcessorList([
                    PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
                ])
                
                generation_config = GenerationConfig(
                    use_cache=False,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                    max_new_tokens=40960,
                    do_sample=True,
                    top_k=2048,
                )
                
                outputs = self.model.generate(
                    inputs.input_ids.to("cuda:0"),
                    generation_config=generation_config,
                    logits_processor=logits_processor,
                    attention_mask=inputs.attention_mask.to("cuda:0"),
                )
                
                decoded_outputs = self.processor.decode(outputs[0])
                response = "".join([str(item) for item in decoded_outputs if not isinstance(item, Image.Image)])
                image_list = [item for item in decoded_outputs if isinstance(item, Image.Image)]

                res_list.append({
                    "query": query,
                    "response": IMAGE_TOKEN(0) + response, # TODO handle multiple image tokens
                    "image_list": image_list,
                    "audio_list": [],
                })

            except Exception as e:
                print(f"Error generating content for query: {query}. Error: {e}")
                res_list.append({
                    "query": query,
                    "response": "<image_start><image_{0}><image_end>",
                    "image_list": [Image.new("RGB", (1024, 1024), "white")],
                    "audio_list": [],
                })

        return res_list
