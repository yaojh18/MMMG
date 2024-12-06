import openai
import time
import base64
import requests
import collections
import re
import numpy as np
import soundfile as sf
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed

OPENAI_KEY = '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'
IMAGE_TOKEN = lambda x: f'<image_start><image_{x}><image_end>'
AUDIO_TOKEN = lambda x: f'<audio_start><audio_{x}><audio_end>'
FAILED_TOKEN = '<none>'
SAMPLE_RATE = 22050


def generate_image_from_openai(index, prompt, model="dall-e-3"):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    retry_count = 2
    retry_interval = 10

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
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, None


def batch_generate_image_from_openai(prompt_list, model_name="dall-e-3"):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_image_from_openai, index, prompt, model_name)
                   for index, prompt in enumerate(prompt_list)]
        image_dict = collections.defaultdict(None)
        for job in tqdm(as_completed(futures), total=len(futures), desc="querying openai..."):
            index, res = job.result(timeout=None)
            image_dict[index] = res

    return [image_dict[i] for i in range(len(prompt_list))]


def encode_image(image: Image.Image, dtype='png'):
    buffer = BytesIO()
    image.save(buffer, format=dtype)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_audio(audio: np.ndarray, dtype='flac'):
    buffer = BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format=dtype)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def form_openai_mm_query(text, images=(), audios=()):
    texts = re.split(r'<(?:image|audio)_start><(?:image|audio)_\d+><(?:image|audio)_end>', text)
    modalities = re.findall(r'<((?:image|audio)_\d+)>', text)
    message = []
    for t, mm in zip(texts[:-1], modalities):
        if t != '':
            message.append({"type": "text", "text": t})
        mm_name, mm_idx = mm.split('_')
        if mm_name == 'image':
            message.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(images[int(mm_idx)])}"
                }
            })
        else:
            message.append({
                "type": "input_audio",
                "input_audio": {
                    "data": encode_audio(audios[int(mm_idx)]),
                    "format": "flac",
                }
            })
    if texts[-1] != '':
        message.append({"type": "text", "text": texts[-1]})
    return [{
        'role': 'user',
        'content': message
    }]


def query_openai(prompt, index, model, max_tokens, temperature):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    retry_count = 2
    retry_interval = 10

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                messages=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
            msg = response.choices[0].message.content
            return index, msg

        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, ''


def batch_query_openai(prompt_list, model_name='gpt-4o-mini', max_new_tokens=768, temperature=0.0):
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(query_openai, prompt, index, model_name, max_new_tokens, temperature)
                   for index, prompt in enumerate(prompt_list)]
        response_dict = collections.defaultdict(str)
        for job in tqdm(as_completed(futures), total=len(futures), desc="querying openai..."):
            index, res = job.result(timeout=None)
            response_dict[index] = res

    return [response_dict[i] for i in range(len(prompt_list))]


def parse_responses(responses, pattern, post_process=lambda x: x):
    pattern = re.compile(pattern, flags=re.IGNORECASE)
    new_responses = []
    for res in responses:
        match = pattern.search(res)
        if match:
            result = match.group(1)
            result = post_process(result)
        else:
            result = FAILED_TOKEN
        new_responses.append(result)
    return new_responses
