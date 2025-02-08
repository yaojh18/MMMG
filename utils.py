import openai
import time
import base64
import requests
import collections
import re
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import torch
import torch.nn.functional as F
from pydub import AudioSegment
from pydub.silence import detect_silence
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from typing import Callable
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import cohen_kappa_score
from transformers import AutoProcessor, ClapModel

OPENAI_KEY = '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'
GEMINI_KEY = 'AIzaSyB-MKMN8fRHpk6LLLR9jrkJfeUxLzX70s8'
HF_KEY = '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'
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


def encode_audio(audio: np.ndarray, dtype='wav', decode=True, return_file=False):
    buffer = BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format=dtype)
    if return_file:
        buffer.seek(0)
        return buffer
    if decode:
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return buffer.getvalue()


def generate_image_from_openai(index, prompt, model="dall-e-3"):
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
    return index, None


def speech_to_text_from_openai(index, audio):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    retry_count = 2
    retry_interval = 1

    for _ in range(retry_count):
        try:
            transcription = client.audio.translations.create(
                model="whisper-1",
                file=encode_audio(audio, return_file=True),
            )
            return index, transcription.text

        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, ''


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
                    "format": "wav",
                }
            })
    if texts[-1] != '':
        message.append({"type": "text", "text": texts[-1]})
    return [{
        'role': 'user',
        'content': message
    }]


def form_gemini_mm_query(text, images=(), audios=()):
    message = [text]
    for audio in audios:
        message.append({
            "mime_type": "audio/wav",
            "data": encode_audio(audio, decode=False)
    })
    return message


def query_openai(index, prompt, model, temperature, dtype='gpt'):
    if dtype == 'gpt':
        client = openai.OpenAI(api_key=OPENAI_KEY)
    elif dtype == 'gemini':
        client = openai.OpenAI(
            api_key=GEMINI_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        raise NotImplementedError
    retry_count = 10
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
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, ''


def query_gemini(index, query, model, temperature):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(model_name=model, generation_config=genai.GenerationConfig(temperature=temperature, top_p=1.0))
    retry_count = 10
    retry_interval = 1

    for _ in range(retry_count):
        try:
            result = model.generate_content(query)
            return index, result.text
        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, ''


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


def calculate_psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ssim(img1, img2, channel_axis=-1)


def calculate_kappa(list1, list2):
    if list1 == list2:
        return 1.0
    return cohen_kappa_score(list1, list2)


def calculate_pearson(list1, list2):
    if list1 == list2:
        return 1.0
    if np.all(np.array(list1) == 0) or np.all(np.array(list2) == 0):
        return 0.0
    return np.corrcoef(list1, list2)[0, 1]


def calculate_agreement(list1, list2):
    print(np.arange(len(list1))[np.array(list1) != np.array(list2)])
    return 1.0 - (np.array(list1) != np.array(list2)).sum() / len(list1)


def color_condition(image: Image.Image, condition: str):
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
    return calculate_ssim(image, ref_image)


def symmetry_condition(image: Image.Image, condition: str):
    if condition == "center":
        ref_image = image.rotate(180)
    elif condition == "horizontal":
        ref_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif condition == "vertical":
        ref_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        raise NotImplementedError

    return calculate_ssim(image, ref_image)


def object_segmentation(image_path):
    from mmdet.apis import init_detector, inference_detector
    config_file = '../mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py'
    checkpoint_file = '../mmdetection/checkpoints/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'
    with open('./data/object_names.txt', 'r') as cls_file:
        classnames = [line.strip() for line in cls_file]
    confidence_threshold = 0.3
    detected = []
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    result = inference_detector(model, image_path).pred_instances
    scores, labels, bboxes = result.scores, result.labels, result.bboxes
    detected_labels = labels[scores >= confidence_threshold]
    detected_bboxes = bboxes[scores >= confidence_threshold]
    for label, bbox in zip(detected_labels, detected_bboxes):
        detected.append((classnames[label], bbox))
    return detected


def generate_ocr_from_gcd(index: int, image: Image.Image, language='zh'):
    """
    Make sure you set confidential first.
    pip install google-cloud-vision
    gcloud init
    gcloud auth application-default login
    """
    from google.cloud import vision
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


def compute_clapscore_at(audio_list, text_list):
    with torch.no_grad():
        audio_list = [librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=48000) for audio in audio_list]
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        inputs = processor(text=text_list, audios=audio_list, return_tensors="pt", padding=True, sampling_rate=48000)
        outputs = model(**inputs)
        cos_sim = F.cosine_similarity(outputs.audio_embeds, outputs.text_embeds)
        return cos_sim.tolist()


def find_optimal_threshold(pred_list, label_list):
    best_threshold = 0.0
    best_accuracy = 0.0
    best_predictions = None

    for threshold in np.linspace(0, 0.99, 100):
        predictions = (pred_list > threshold).astype(int)
        accuracy = (predictions == label_list).mean()
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_predictions = predictions
    print('Best threshold: ', best_threshold)

    return best_threshold


def compute_clapscore_aa(audio, ref_audio_list):
    with torch.no_grad():
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=48000)
        ref_audio_list = [librosa.resample(ref_audio, orig_sr=SAMPLE_RATE, target_sr=48000) for ref_audio in ref_audio_list]
        ref_audio_list.append(audio)
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        audio_inputs = processor(audios=ref_audio_list, sampling_rate=48000, return_tensors="pt", padding=True)
        audio_embeddings = model.get_audio_features(**audio_inputs)
        cos_sim = F.cosine_similarity(audio_embeddings[-1], audio_embeddings[:-1])
        return float(cos_sim.topk(10)[0].mean())


def audio_classification(audio_list, label_list):
    from models.beats.BEATs import BEATs, BEATsConfig
    checkpoint = torch.load('./models/beats/checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')
    pred_map = pd.read_csv('./datasets/ESC-50/class_labels_indices.csv')
    pred_map.set_index('display_name', inplace=True)
    inverted_label_dict = {v: k for k, v in checkpoint['label_dict'].items()}
    label_list = pred_map.loc[label_list]['mid'].tolist()
    label_list = torch.tensor([inverted_label_dict[label] for label in label_list])
    with torch.no_grad():
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()
        probs = BEATs_model.extract_features(torch.tensor(audio_list))[0]
        predictions = probs.gather(dim=-1, index=label_list.unsqueeze(1)).squeeze(1).tolist()
        return predictions


def audio_segmentation(audio, top_db=60, min_duration=1.0):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    segments = []
    min_silence_samples = int(min_duration * SAMPLE_RATE)
    previous_end = non_silent_intervals[0][0]

    for i in range(len(non_silent_intervals) - 1):
        if non_silent_intervals[i][1] + min_silence_samples < non_silent_intervals[i + 1][0]:
            segments.append(audio[previous_end: non_silent_intervals[i][1]])
            previous_end = non_silent_intervals[i + 1][0]
    if previous_end < non_silent_intervals[-1][1]:
        segments.append(audio[previous_end:])
    return segments
