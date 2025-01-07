import openai
import time
import base64
import requests
import collections
import re
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageChops
from io import BytesIO
from typing import Callable
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.cloud import vision

OPENAI_KEY = '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'
IMAGE_TOKEN = lambda x: f'<image_start><image_{x}><image_end>'
AUDIO_TOKEN = lambda x: f'<audio_start><audio_{x}><audio_end>'
FAILED_TOKEN = '<none>'
SAMPLE_RATE = 22050

idx2letter = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
letter2idx = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25
}


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


def batch(func_name: Callable, data_list, num_worker=4, **kwargs):
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures = [executor.submit(func_name, index, data, **kwargs) for index, data in enumerate(data_list)]
        res_dict = collections.defaultdict(None)
        for job in tqdm(as_completed(futures), total=len(futures), desc="Working..."):
            index, res = job.result(timeout=None)
            res_dict[index] = res

    return [res_dict[i] for i in range(len(data_list))]


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


def query_openai(index, prompt, model, temperature):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    retry_count = 2
    retry_interval = 10

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                messages=prompt,
                model=model,
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


def generate_ocr_from_gcd(index: int, image: Image.Image, language='zh'):
    """
    Make sure you set confidential first.
    pip install google-cloud-vision
    gcloud init
    gcloud auth application-default login
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format='png')
    image = vision.Image(content=image_bytes.getvalue())

    client = vision.ImageAnnotatorClient()
    retry_count = 2
    retry_interval = 10

    for _ in range(retry_count):
        try:
            response = client.text_detection(image=image, image_context={"language_hints": [language]}, )
            if response.error.message:
                raise Exception(response.error.message)
            results = []
            for text in response.text_annotations:
                vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                results.append({"text": text.description, "box": vertices})
            return index, results
        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, []


def calculate_f1(text1, text2):
    tokens1 = text1.split(' ')
    tokens2 = text2.split(' ')
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)

    overlap = sum((counter1 & counter2).values())
    precision = overlap / sum(counter2.values()) if counter2 else 0
    recall = overlap / sum(counter1.values()) if counter1 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def color_condition_range(image: Image.Image, condition: str):
    # cond_func = {
    #     "green": lambda color: 75 <= color[0] <= 195,
    #     "blue": lambda color: 175 <= color[0] <= 260,
    #     "yellow": lambda color: 20 <= color[0] <= 70,
    #     "gray": lambda color: color[1] <= 15 or color[2] <= 15,
    #     "pink": lambda color: (0 <= color[0] <= 15 or 280 <= color[0] <= 360) and color[1] <= 85 and color[2] >= 35,
    #     "red": lambda color: (0 <= color[0] <= 30 or 320 <= color[0] <= 360),
    #     "orange": lambda color: 0 <= color[0] <= 50,
    #     "purple": lambda color: 220 <= color[0] <= 310,
    #     "cyan": lambda color: 155 <= color[0] <= 215,
    # }[condition]

    cond_func = {
        "green": lambda color: 80 <= color[0] <= 170,
        "blue": lambda color: 180 <= color[0] <= 260,
        "yellow": lambda color: 30 <= color[0] <= 80,
        "gray": lambda color: color[1] <= 15 or color[2] <= 15,
        "pink": lambda color: (0 <= color[0] <= 10 or 280 <= color[0] <= 360) and color[1] <= 75 and color[2] >= 50,
        "red": lambda color: (0 <= color[0] <= 30 or 330 <= color[0] <= 360),
        "orange": lambda color: 10 <= color[0] <= 50,
        "purple": lambda color: 240 <= color[0] <= 310,
        "cyan": lambda color: 150 <= color[0] <= 210,
    }[condition]

    image = image.convert('HSV')
    img_arr = np.array(image)
    img_arr = np.dot(img_arr, np.diag([360 / 255, 100 / 255, 100 / 255]))
    img_cond = np.apply_along_axis(cond_func, axis=-1, arr=img_arr)
    # black_or_white = np.apply_along_axis(lambda color: color[1] <= 15 or color[2] <= 15, axis=-1, arr=img_arr) if condition != 'gray' else np.zeros_like(img_cond)
    # final_score = np.where(img_cond, 1, -10) + np.where(black_or_white, 10, 0)
    # final_score = np.clip(final_score, a_min=None, a_max=1).reshape(img_cond.shape) + np.where(black_or_white, -1, 0)

    # visualize for debugging
    # plt.imshow(img_cond > 0, cmap='gray', interpolation='nearest')
    # plt.axis('off')
    # plt.show()

    # return float(final_score.mean()) / 2 + 0.5
    return img_cond.mean()


def color_condition_exact(image: Image.Image, condition: str):
    color = {
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "pink": (255, 128, 255),
        "red": (255, 0, 0),
        "orange": (255, 128, 0),
        "purple": (128, 0, 128),
        "cyan": (0, 255, 255),
    }[condition]
    ref_image = Image.new("RGB", image.size, color)
    diff = ImageChops.difference(image, ref_image)
    return (np.array(diff).mean(axis=-1) < 25.6).mean()


def symmetry_condition(image: Image.Image, condition: str):
    if condition == "center":
        rotated = image.rotate(180)
        diff = ImageChops.difference(image, rotated)
    elif condition == "horizontal":
        width, height = image.size
        top_half = image.crop((0, 0, width, height // 2))
        bottom_half = image.crop((0, height // 2, width, height))
        bottom_half_flipped = bottom_half.transpose(Image.FLIP_TOP_BOTTOM)
        diff = ImageChops.difference(top_half, bottom_half_flipped)
    elif condition == "vertical":
        width, height = image.size
        left_half = image.crop((0, 0, width // 2, height))
        right_half = image.crop((width // 2, 0, width, height))
        right_half_flipped = right_half.transpose(Image.FLIP_LEFT_RIGHT)
        diff = ImageChops.difference(left_half, right_half_flipped)
    else:
        raise NotImplementedError

    return (np.array(diff).mean(axis=-1) < 25.6).mean()


# def object_segmentation(image_path):
#     config_file = '../mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py'
#     checkpoint_file = '../mmdetection/checkpoints/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'
#     with open('./data/object_names.txt', 'r') as cls_file:
#         classnames = [line.strip() for line in cls_file]
#     confidence_threshold = 0.3
#     detected = []
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#     result = inference_detector(model, image_path).pred_instances
#     scores, labels, bboxes = result.scores, result.labels, result.bboxes
#     detected_labels = labels[scores >= confidence_threshold]
#     detected_bboxes = bboxes[scores >= confidence_threshold]
#     for label, bbox in zip(detected_labels, detected_bboxes):
#         detected.append((classnames[label], bbox))
#     return detected
