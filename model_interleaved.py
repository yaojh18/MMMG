import re
import sys
import itertools

from torchvision.models.detection import image_list

from model import *
from model_image import *
from model_audio import *
from utils import *
from prompt import *


### Agent models
class AudioAgent(Model):
    sound_model_name = 'BlankAudioModel'
    speech_model_name = 'VoxInstruct'
    music_model_name = 'BlankAudioModel'

    def __init__(self, mllm='gemini-2.0-flash'):
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
            cnt = 0
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
                res = res.replace(audio_prompts[i], f"<audio_{cnt}>")
                cnt += 1
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
        if os.path.exists('./temp/AudioAgent/'):
            shutil.rmtree('./temp/AudioAgent/')
        os.makedirs('./temp/AudioAgent/')
        for output in output_list:
            audio_map = {}
            for i in range(len(output['audio_list'])):
                if isinstance(output['audio_list'][i], tuple):
                    audio_map[i] = output['audio_list'][i]
                    output['audio_list'][i] = responses[audio_map[i][0]][audio_map[i][1]]
                    sf.write(f"./temp/AudioAgent/{audio_map[i][0]}_{audio_map[i][1]}.wav", output['audio_list'][i], SAMPLE_RATE)
                elif isinstance(output['audio_list'][i], dict):
                    ref_id = output['audio_list'][i]['reference']
                    output['audio_list'][i]['reference'] = f"./temp/AudioAgent/{audio_map[ref_id][0]}_{audio_map[ref_id][1]}.wav"
                    tts_query_list.append(output['audio_list'][i])
                    output['audio_list'][i] = idx
                    idx += 1
        if len(tts_query_list) > 0:
            responses = self.models[1].generate(tts_query_list)
            responses = [res['audio_list'][0] for res in responses]
            for output in output_list:
                for i in range(len(output['audio_list'])):
                    if isinstance(output['audio_list'][i], int):
                        output['audio_list'][i] = responses[output['audio_list'][i]]
        for output in output_list:
            output['audio_list'] = [a for a in output['audio_list'] if a is not FAILED_TOKEN]
        return output_list


class VoxInstructAgent(AudioAgent):
    speech_model_name = 'VoxInstruct'


class VoiceLDMAgent(AudioAgent):
    speech_model_name = 'VoiceLDM'


class ImageAgent(Model):
    mllm_name: str
    diffusion_name: str

    def __init__(self):
        if 'gpt' in self.mllm_name:
            self.mllm = OpenAIModel(self.mllm_name, system_prompt=I_AGENT_PROMPT)
        elif 'gemini' in self.mllm_name:
            self.mllm = GeminiModel(self.mllm_name, system_prompt=I_AGENT_PROMPT)
        else:
            raise NotImplementedError
        self.diffusion = eval(f'{self.diffusion_name}()')

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
            output['image_list'] = list(itertools.chain(*[res_list[i]['image_list'] for i in output['image_list']]))
        return output_list


class GPTAgent(ImageAgent):
    mllm_name = 'gpt-4o'
    diffusion_name = 'Dalle3'


class GeminiAgent(ImageAgent):
    mllm_name = 'gemini-2.0-flash'
    diffusion_name = 'Imagen3'


class ImageAllAgent(Model):
    mllm_name: str
    diffusion_name: str

    def __init__(self):
        if 'gpt' in self.mllm_name:
            self.mllm = OpenAIModel(self.mllm_name, system_prompt=I_ALL_AGENT_PROMPT)
        elif 'gemini' in self.mllm_name:
            self.mllm = GeminiModel(self.mllm_name, system_prompt=I_ALL_AGENT_PROMPT)
        else:
            raise NotImplementedError
        self.diffusion = eval(f'{self.diffusion_name}()')

    def generate(self, query_list):
        responses = self.mllm.generate(query_list)
        output_list = []
        pattern = r'<image_start>(.*?)</?image_end>'
        image_pattern = r'<[\s/]*image_prompt="(.*?)"[\s/]*><[\s/]*image_ref=\[(.*?)\][\s/]*>'
        for query, res in zip(query_list, responses):
            image_prompts = re.findall(pattern, res)
            res = res.replace('</image_end>', '<image_end>')
            image_list = []
            cnt = 0
            for i in range(len(image_prompts)):
                image_prompt = re.match(image_pattern, image_prompts[i])
                if image_prompt is None:
                    image_list.append(FAILED_TOKEN)
                    continue
                image_prompt = image_prompt.groups()
                if image_prompt[1] == '':
                    image_list.append({'type': 'gen', 'prompt': image_prompt[0], 'reference': []})
                else:
                    ref_list = image_prompt[1].split(',')
                    if all(s.isnumeric() and 0 <= int(s) < len(query['image_list']) for s in ref_list):
                        ref_list = [int(s) for s in ref_list]
                        image_list.append({'type': 'edit', "prompt": image_prompt[0], 'reference': ref_list})
                    elif all(len(s) > 1 and s[0] == '#' and s[1:].isnumeric() and 0 <= int(s[1:]) < len(image_list) for s in ref_list):
                        ref_list = [int(s[1:]) for s in ref_list]
                        image_list.append({'type': 'edit_gen', 'prompt': image_prompt[0], 'reference': ref_list})
                    else:
                        image_list.append(FAILED_TOKEN)
                        continue
                res = res.replace(image_prompts[i], f'<image_{cnt}>')
                cnt += 1
            output_list.append({
                'query': query,
                'response': res,
                'image_list': image_list,
                'audio_list': [],
            })
        diffusion_query_list = []
        idx = 0
        for output in output_list:
            for i in range(len(output['image_list'])):
                if output['image_list'][i] != FAILED_TOKEN and output['image_list'][i]['type'] != 'edit_gen':
                    diffusion_query_list.append({
                        'instruction': output['image_list'][i]['prompt'],
                        'image_list': [output['query']['image_list'][j] for j in output['image_list'][i]['reference']]
                    })
                    output['image_list'][i] = idx
                    idx += 1
        while len(diffusion_query_list) > 0:
            ## Store this turn
            responses = self.diffusion.generate(diffusion_query_list)
            responses = [res['image_list'][0] if len(res['image_list']) > 0 else
                         Image.new('RGB', (1024, 1024), color='white') for res in responses]
            for output in output_list:
                for i in range(len(output['image_list'])):
                    if isinstance(output['image_list'][i], int):
                        output['image_list'][i] = responses[output['image_list'][i]]

            ## Generate query for next turn
            if os.path.exists('./temp/ImageAgent/'):
                shutil.rmtree('./temp/ImageAgent/')
            os.makedirs('./temp/ImageAgent/')
            diffusion_query_list = []
            idx = 0
            img_idx = 0
            for output in output_list:
                image_map = {}
                for i in range(len(output['image_list'])):
                    if (isinstance(output['image_list'][i], dict) and
                            all(isinstance(output['image_list'][j], Image.Image) for j in output['image_list'][i]['reference'])):
                        for j in output['image_list'][i]['reference']:
                            if j not in image_map:
                                image_map[j] = img_idx
                                output['image_list'][j].save(f'./temp/ImageAgent/{img_idx}.png')
                                img_idx += 1
                        diffusion_query_list.append({
                            'instruction': output['image_list'][i]['prompt'],
                            'image_list': [f'./temp/ImageAgent/{image_map[j]}.png' for j in output['image_list'][i]['reference']]
                        })
                        output['image_list'][i] = idx
                        idx += 1

        for output in output_list:
            output['image_list'] = [a if isinstance(a, Image.Image) else Image.new('RGB', (1024, 1024), color='white')
                                    for a in output['image_list'] if a is not FAILED_TOKEN]
        return output_list


class GPT4oAgent(ImageAllAgent):
    mllm_name = 'gpt-4o'
    diffusion_name = 'GPT4o'


class MultiTurnAgent(Model):
    modality = 'image' or 'audio'
    model_name: str

    def __init__(self):
        self.system_prompt = I_MULTI_TURN_AGENT_PROMPT if self.modality == 'image' else A_MULTI_TURN_AGENT_PROMPT
        self.model = eval(f'{self.model_name}()')

    def apply_chat_template(self, index, query, his_list):
        query = {
            'instruction': f"{self.system_prompt}\n<|im_start|>user\n{query['instruction']}\n",
            'image_list': query['image_list'] if 'image_list' in query else [],
            'audio_list': query['audio_list'] if 'audio_list' in query else []
        }
        query['instruction'] += ''.join([IMAGE_TOKEN(i) + '\n' for i, _ in enumerate(query['image_list'])])
        query['instruction'] += ''.join([AUDIO_TOKEN(i) + '\n' for i, _ in enumerate(query['audio_list'])])
        query['instruction'] += '<|im_end|>\n<|im_start|>assistant\n'
        mm_cnt = 0
        for his in his_list:
            if his is None:
                break
            elif isinstance(his, str):
                query['instruction'] += his + '\n<|im_end|>\n<|im_start|>user\nplease continue\n<|im_end|>\n<|im_start|>assistant\n'
            else:
                query['instruction'] += ((IMAGE_TOKEN(mm_cnt) if self.modality == 'image' else AUDIO_TOKEN(mm_cnt))
                                         + '\n<|im_end|>\n<|im_start|>user\nplease continue\n<|im_end|>\n<|im_start|>assistant\n')
                if self.modality == 'image':
                    his.save(f'./temp/ImageAgent/{index}_{mm_cnt}.png')
                    query['image_list'].append(f'./temp/ImageAgent/{index}_{mm_cnt}.png')
                else:
                    sf.write(f'./temp/AudioAgent/{index}_{mm_cnt}.png', his, SAMPLE_RATE)
                    query['audio_list'].append(f'./temp/AudioAgent/{index}_{mm_cnt}.png')
                mm_cnt += 1
        return query

    @staticmethod
    def remove_template(text):
        return (text.replace('<|im_start|>user', '').replace('<|im_start|>system', '').replace('<|im_start|>assistant', '')
                .replace('<|im_start|>', '').replace('<|im_end|>', '').replace('please continue', '').replace('<|file_separator|>', '').strip())

    def generate(self, query_list):
        turn_query_list = [self.apply_chat_template(0, query, []) for query in query_list]
        res_idx_list = [i for i in range(len(query_list))]
        res_list = [[] for i in range(len(query_list))]

        turn_cnt = 0
        pattern = r'(<(?:image|audio)_start><(?:image|audio)_\d+><(?:image|audio)_end>|<stop>)'
        while len(turn_query_list) > 0 and turn_cnt < 10:
            # Generate and store responses
            responses = self.model.generate(turn_query_list)
            for i in range(len(query_list)):
                if res_idx_list[i] is not None:
                    res = responses[res_idx_list[i]]
                    segments = re.split(pattern, res['response'])
                    segments = [self.remove_template(segment) for segment in segments if self.remove_template(segment)]
                    for segment in segments:
                        if segment == '<stop>':
                            res_list[i].append(None)
                            break
                        elif self.modality == 'image' and re.match('<image_start><image_\d+><image_end>', segment):
                            idx = int(re.match('<image_start><image_(\d+)><image_end>', segment).group(1))
                            if 0 <= idx < len(res['image_list']):
                                res_list[i].append(res['image_list'][idx])
                        elif self.modality == 'audio' and re.match('<audio_start><audio_\d+><audio_end>', segment):
                            idx = int(re.match('<audio_start><audio_(\d+)><audio_end>', segment).group(1))
                            if 0 <= idx < len(res['audio_list']):
                                res_list[i].append(res['audio_list'][idx])
                        else:
                            res_list[i].append(segment)

            # Generate query for next turn
            if os.path.exists('./temp/ImageAgent/'):
                shutil.rmtree('./temp/ImageAgent/')
            os.makedirs('./temp/ImageAgent/')
            turn_query_list = []
            res_idx_list = []
            idx = 0
            for query, res in zip(query_list, res_list):
                if res[-1] is not None:
                    turn_query_list.append(self.apply_chat_template(idx, query, res))
                    res_idx_list.append(idx)
                    idx += 1
                else:
                    res_idx_list.append(None)
            turn_cnt += 1

        output_list = []
        for query, res in zip(query_list, res_list):
            mm_list = []
            response = ''
            for r in res[:-1] if res[-1] is None else res:
                if isinstance(r, str):
                    response += r + '\n'
                else:
                    response += IMAGE_TOKEN(len(mm_list)) if self.modality == 'image' else AUDIO_TOKEN(len(mm_list))
                    mm_list.append(r)
            output_list.append({
                'query': query,
                'response': response,
                'image_list': mm_list if self.modality == 'image' else [],
                'audio_list': mm_list if self.modality == 'audio' else []
            })
        return output_list

### Interleaved I+T model


class Gemini2(Model):
    model_name = 'gemini-2.0-flash-exp-image-generation'
    system_prompt = IT_AGENT_PROMPT

    def __init__(self):
        super().__init__()

    def generate(self, query_list):
        client = genai.Client(api_key=GEMINI_KEY)
        res_list = []

        for query in tqdm(query_list):
            retry_count = 3
            retry_interval = 10
            flag = False
            for _ in range(retry_count):
                try:
                    # contents = [f'## System Prompt: \n{self.system_prompt}\n ## User prompt: \n' + query['instruction']]
                    contents = [query['instruction']]
                    images = query.get("image_list", [])
                    for img_path in images:
                        contents.append(Image.open(img_path))
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["Text", "Image"],
                            temperature=0.1,
                            top_p=1.0
                        ),
                    )
                    generated_text = ""
                    generated_images = []
                    image_count = 0
                    for part in response.candidates[0].content.parts:
                        if part.text is not None:
                            generated_text += part.text
                        if part.inline_data is not None:
                            generated_images.append(Image.open(BytesIO(part.inline_data.data)))
                            generated_text += IMAGE_TOKEN(image_count)
                            image_count += 1

                    res_list.append({
                        "query": query,
                        "response": generated_text,
                        "image_list": generated_images,
                        "audio_list": [],
                    })
                    flag = True
                    break

                except Exception as e:
                    print(f"Error processing query: {query}. Error: {e}")
                    time.sleep(retry_interval)
                    retry_interval *= 2

            if not flag:
                res_list.append({
                    "query": query,
                    "response": '',
                    "image_list": [],
                    "audio_list": [],
                })

        return res_list


class Emu3(Model):
    def __init__(self):
        super().__init__()

        from models.emu3.mllm.processing_emu3 import Emu3Processor
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        EMU_HUB = "BAAI/Emu3-Gen"
        VQ_HUB = "BAAI/Emu3-VisionTokenizer"

        self.model = AutoModelForCausalLM.from_pretrained(
            EMU_HUB,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            token=HF_KEY,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left", token=HF_KEY)
        self.image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, token=HF_KEY)
        self.image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda", trust_remote_code=True, token=HF_KEY).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)
        self.system_prompt = IT_AGENT_PROMPT

    def generate(self, query_list):
        from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
        from transformers.generation.configuration_utils import GenerationConfig

        res_list = []
        for query in tqdm(query_list):
            try:
                instruction = query.get("instruction", "")
                images = query.get("image_list", [])

                inputs = self.processor(
                    text=f'## System Prompt: \n{self.system_prompt}\n ## User prompt: \n' + instruction,
                    images=images,
                    mode="G",
                    ratio="1:1",
                    image_area=self.model.config.image_area,
                    return_tensors="pt",
                    padding="longest",
                )

                h = inputs.image_size[:, 0]
                w = inputs.image_size[:, 1]
                constrained_fn = self.processor.build_prefix_constrained_fn(h, w)

                logits_processor = LogitsProcessorList([
                    UnbatchedClassifierFreeGuidanceLogitsProcessor(
                        3.0,
                        self.model,
                    ),
                    PrefixConstrainedLogitsProcessor(
                        constrained_fn,
                        num_beams=1,
                    ),
                ])

                generation_config = GenerationConfig(
                    use_cache=True,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                    max_new_tokens=40960,
                    do_sample=True,
                    temperature=0.1,
                    top_p=1.0
                )

                outputs = self.model.generate(
                    inputs.input_ids.to("cuda"),
                    generation_config=generation_config,
                    logits_processor=logits_processor,
                    attention_mask=inputs.attention_mask.to("cuda"),
                )

                decoded_outputs = self.processor.decode(outputs[0])

                parts, image_list = [], []
                image_count = 0
                for item in decoded_outputs:
                    if isinstance(item, Image.Image):
                        token = IMAGE_TOKEN(image_count)
                        parts.append(token)
                        image_list.append(item)
                        image_count += 1
                    else:
                        parts.append(str(item))

                response = "".join(parts)

                res_list.append({
                    "query": query,
                    "response": response,
                    "image_list": image_list,
                    "audio_list": [],
                })

            except Exception as e:
                print(f"Error generating content for query: {query}. Error: {e}")
                res_list.append({
                    "query": query,
                    "response": "",
                    "image_list": [],
                    "audio_list": [],
                })

        return res_list
