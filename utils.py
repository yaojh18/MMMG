import os
import time
import base64
import collections
import re
import librosa
import json
import colorsys
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from typing import Callable
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor, as_completed


OPENAI_KEY = os.getenv('OPENAI_KEY')
GEMINI_KEY = os.getenv('GEMINI_KEY')
REPLICATE_KEY = os.getenv('REPLICATE_KEY')
RECRAFT_KEY = os.getenv('RECRAFT_KEY')
HF_KEY = os.getenv('HF_KEY')
IMAGE_TOKEN = lambda x: f'<image_start><image_{x}><image_end>'
AUDIO_TOKEN = lambda x: f'<audio_start><audio_{x}><audio_end>'
FAILED_TOKEN = '<none>'
SAMPLE_RATE = 22050
VISION_MODEL = 'openai'


def batch(func_name: Callable, data_list, num_worker=8, **kwargs):
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


def decode_image(encoded_image: str):
    image_data = base64.b64decode(encoded_image)
    image_buffer = BytesIO(image_data)
    return Image.open(image_buffer)


def encode_audio(audio: np.ndarray, dtype='wav', decode=True, return_file=False):
    buffer = BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format=dtype)
    if return_file:
        buffer.seek(0)
        return buffer
    if decode:
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return buffer.getvalue()


def form_mm_query(text, images=[], audios=[], model=''):
    model = model or VISION_MODEL
    if model == 'gemini':
        return form_gemini_mm_query(text, images, audios)
    elif model == 'openai':
        return form_openai_mm_query(text, images, audios)
    elif model == 'qwen':
        return form_qwen_mm_query(text, images, audios)
    else:
        raise NotImplementedError('Vision model not implemented.')


def form_openai_mm_query(text, images=[], audios=[]):
    message = []
    for image in images:
        message.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(image)}"
            }
        })
    for audio in audios:
        message.append({
            "type": "input_audio",
            "input_audio": {
                "data": encode_audio(audio),
                "format": "wav",
            }
        })
    message.append({"type": "text", "text": text})
    return [{
        'role': 'user',
        'content': message
    }]


def form_gemini_mm_query(text, images=[], audios=[]):
    from google.genai import types
    message = [text] + images
    for audio in audios:
        message.append(types.Part.from_bytes(
            data=encode_audio(audio, decode=False),
            mime_type='audio/wav',
        ))
    return message


def form_qwen_mm_query(text, images=[], audios=[]):
    """
    Qwen2.5 doesn't support audio input for now. So we just dismiss audios.
    """
    message = [{"type": "text", "text": text}]
    for image in images:
        message.append({"type": "image", "image": f"data:image/png;base64,{encode_image(image)}"})
    return [{
        'role': 'user',
        'content': message
    }]


def query_vlm(query_list, model=''):
    model = model or VISION_MODEL
    if model == 'gemini':
        return batch(query_gemini, query_list, model='gemini-2.5-pro-preview-03-25', temperature=0.0)
    elif model == 'openai':
        return batch(query_openai, query_list, model='chatgpt-4o-latest', temperature=0.0)
    elif model == 'qwen':
        return batch_query_qwen(query_list, temperature=0.0)
    else:
        raise NotImplementedError('Vision model not implemented.')


def query_openai(index, prompt, model, temperature):
    import openai
    client = openai.OpenAI(api_key=OPENAI_KEY)
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
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)
    retry_count = 10
    retry_interval = 10

    for _ in range(retry_count):
        try:
            result = client.models.generate_content(
                model=model, contents=query,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=1.0
                )
            )
            if result.text is None:
                return index, ''
            return index, result.text
        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response.')
    return index, ''


def batch_query_qwen(query_list, temperature):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    output_list = []
    generation_kwargs = {'max_new_tokens': 256}
    if temperature == 0.0:
        generation_kwargs['do_sample'] = False
    else:
        generation_kwargs['temperature'] = temperature
        generation_kwargs['top_p'] = 1.0
        generation_kwargs['do_sample'] = True

    for query in tqdm(query_list):
        text = processor.apply_chat_template(
            query, tokenize=False, add_generation_prompt=True, use_fast=True
        )
        image_inputs, video_inputs = process_vision_info(query)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_list.append(processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0])
    return output_list


def calculate_ssim(img1, img2):
    from skimage.metrics import structural_similarity as ssim
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.LANCZOS)
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ssim(img1, img2, channel_axis=-1)


dreamsim_model = None
def calculate_dreamsim(img1, img2):
    from dreamsim import dreamsim
    def preprocess(img):
        img = img.convert('RGB')
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])(img).unsqueeze(0)

    global dreamsim_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dreamsim_model is None:
        dreamsim_model, _ = dreamsim(pretrained=True, device=device, cache_dir="./libs/DreamSim")
    img1 = preprocess(img1).to(device)
    img2 = preprocess(img2).to(device)
    return 1.0 - float(dreamsim_model(img1, img2))


def release_dreamsim():
    global dreamsim_model
    if dreamsim_model is not None:
        del dreamsim_model
        dreamsim_model = None


def calculate_pearson(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    if np.all(list1 == list2):
        return 1.0
    if np.std(list1) == 0:
        list1 += np.random.normal(0, 1e-8, list1.shape)
    if np.std(list2) == 0:
        list2 += np.random.normal(0, 1e-8, list2.shape)
    return np.corrcoef(list1, list2)[0, 1]


def calculate_agreement(list1, list2, return_list=False):
    if return_list:
        return np.arange(len(list1))[np.array(list1) != np.array(list2)]
    return (np.array(list1) == np.array(list2)).sum() / len(list1)


def color_condition(image: Image.Image, condition: str):
    img_array = np.array(image)
    avg_color = tuple(np.mean(img_array.reshape(-1, 3), axis=0).astype(int))
    color = np.array({
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
    }[condition])
    avg_color_hsv = colorsys.rgb_to_hsv(*avg_color)
    color_hsv = colorsys.rgb_to_hsv(*color)
    if condition == "white" or condition == "black":
        if np.linalg.norm(color - avg_color) > 38.4:
            return 0.0, avg_color
    else:
        if 0.15 < abs(avg_color_hsv[0] - color_hsv[0]) < 0.85:
            return 0.0, avg_color
    ref_image = Image.new("RGB", image.size, avg_color)
    return calculate_ssim(image, ref_image), avg_color


def count_pixels(image, reference_color, max_distance=4):
    image = np.array(image)
    ref_color = np.array(reference_color)
    max_distance_squared = max_distance ** 2
    pixels = image.reshape(-1, image.shape[-1])
    squared_distances = np.sqrt(np.sum((pixels - ref_color) ** 2, axis=1))
    count = np.sum(squared_distances <= max_distance_squared)
    return count / (image.shape[0] * image.shape[1])


def compute_clapscore_at(audio_list, text_list):
    from transformers import ClapModel, AutoProcessor
    with torch.no_grad():
        audio_list = [librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=48000) for audio in audio_list]
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to('cuda')
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        inputs = processor(text=text_list, audios=audio_list, return_tensors="pt", padding=True, sampling_rate=48000).to('cuda')
        outputs = model(**inputs)
        cos_sim = F.cosine_similarity(outputs.audio_embeds, outputs.text_embeds)
        return cos_sim.tolist()


def compute_clapscore_aa(audio, ref_audio_list):
    from transformers import ClapModel, AutoProcessor
    with torch.no_grad():
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=48000)
        ref_audio_list = [librosa.resample(ref_audio, orig_sr=SAMPLE_RATE, target_sr=48000) for ref_audio in ref_audio_list]
        ref_audio_list.append(audio)
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to('cuda')
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        audio_inputs = processor(audios=ref_audio_list, sampling_rate=48000, return_tensors="pt", padding=True).to('cuda')
        audio_embeddings = model.get_audio_features(**audio_inputs)
        cos_sim = F.cosine_similarity(audio_embeddings[-1], audio_embeddings[:-1])
        cos_sim, topk = cos_sim.topk(10)
        # print(topk, cos_sim.mean())
        return float(cos_sim.mean())


def find_optimal_threshold(pred_list, label_list):
    best_threshold = 0.0
    best_accuracy = 0.0
    for threshold in np.linspace(min(pred_list) - 0.1, max(pred_list) + 0.1, 100):
        predictions = (pred_list > threshold).astype(int)
        accuracy = (predictions == label_list).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    print('Best threshold: ', best_threshold)

    return best_threshold


def find_optimal_thresholds(pred_list, label_list):
    best_low_threshold = 0.0
    best_high_threshold = 1.0
    best_accuracy = 0.0
    min_gap = max(pred_list) - min(pred_list) + 0.2
    print(min(pred_list), max(pred_list))
    low_thresholds = np.linspace(min(pred_list) - 0.1, max(pred_list), 100)
    high_thresholds = np.linspace(min(pred_list), max(pred_list) + 0.1, 100)

    for low_threshold in low_thresholds:
        for high_threshold in high_thresholds:
            if low_threshold >= high_threshold:
                continue

            predictions = np.zeros_like(pred_list)
            predictions[pred_list > high_threshold] = 2
            predictions[(pred_list >= low_threshold) & (pred_list <= high_threshold)] = 1

            accuracy = (predictions == label_list).mean()
            gap = high_threshold - low_threshold
            if accuracy >= best_accuracy or accuracy == best_accuracy and gap < min_gap:
                best_accuracy = accuracy
                min_gap = gap
                best_low_threshold = low_threshold
                best_high_threshold = high_threshold

    print(f'Best low threshold: {best_low_threshold}, Best high threshold: {best_high_threshold}')

    return best_low_threshold, best_high_threshold


def audio_segmentation(audio, top_db=40, min_duration=1.0):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    segments = []
    min_frames = int(min_duration * SAMPLE_RATE)
    previous_end = non_silent_intervals[0][0]
    for i in range(len(non_silent_intervals) - 1):
        if non_silent_intervals[i][1] + min_frames < non_silent_intervals[i + 1][0]:
            segments.append(audio[previous_end: non_silent_intervals[i][1]])
            previous_end = non_silent_intervals[i + 1][0]
    if previous_end < non_silent_intervals[-1][1]:
        segments.append(audio[previous_end:])
    return segments


def transcribe_speech(audio_list, text_list=None, language='english'):
    import evaluate
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("BELLE-2/Belle-whisper-large-v3-zh" if language == 'chinese' else "openai/whisper-large-v3")
    wer = evaluate.load('cer') if language == 'chinese' else evaluate.load('wer')
    trans_list = []
    wer_list = []
    if text_list is None:
        text_list = [None] * len(audio_list)

    for audio, text in zip(audio_list, text_list):
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt", language=language).input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features, language=language)[0]
            transcription = processor.decode(predicted_ids, language=language)
        transcription = processor.tokenizer._normalize(transcription)
        trans_list.append(transcription)
        if text is not None:
            text = processor.tokenizer._normalize(text)
            if language == 'chinese':
                text = text.replace(' ', '')
            wer_list.append(max(1.0 - wer.compute(references=[text], predictions=[transcription]), 0.0))
    return trans_list, wer_list


def calculate_pitch(audio, gender=None, inst=''):
    import parselmouth
    def extract_pitch(audio, hop_size=256, f0_min=80, f0_max=600, num_bins=100):
        pitch_obj = parselmouth.Sound(audio, SAMPLE_RATE).to_pitch(
            time_step=hop_size / SAMPLE_RATE,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        pitch_values = pitch_obj.selected_array['frequency']
        pitch_times = pitch_obj.xs()
        intensity_obj = parselmouth.Sound(audio, SAMPLE_RATE).to_intensity(
            time_step=hop_size / SAMPLE_RATE, minimum_pitch=f0_min
        )
        intensity_values = intensity_obj.values.flatten()
        intensity_times = intensity_obj.xs()

        if len(pitch_times) != len(intensity_times):
            intensity_values = np.interp(pitch_times, intensity_times, intensity_values)
        voiced_mask = pitch_values > 0.0
        pitch_values = pitch_values[voiced_mask]
        intensity_values = intensity_values[voiced_mask]

        mel_pitch_values = 2595 * np.log10(1 + pitch_values / 700)
        bins = np.linspace(min(mel_pitch_values) - 0.1, max(mel_pitch_values) + 0.1, num=num_bins)
        bin_indices = np.digitize(mel_pitch_values, bins)
        cumulative_intensity = np.zeros(num_bins)
        for idx, val in zip(bin_indices, intensity_values):
            cumulative_intensity[idx] += val
        best_bin = np.argmax(cumulative_intensity)
        best_mask = (bin_indices == best_bin)
        main_mel_pitch = np.sum(mel_pitch_values[best_mask] * intensity_values[best_mask]) / np.sum(intensity_values[best_mask])
        return main_mel_pitch

    pitch = extract_pitch(audio)

    if 'pitch' in inst:
        if gender == 0:
            pitch_s = max(140.0, min(182.0, pitch))
            pitch_s = float(inst['pitch'] == 'high') * (pitch_s - 140.0) / 42.0 + float(inst['pitch'] == 'low') * (182.0 - pitch_s) / 42.0
        else:
            pitch_s = max(236.0, min(278.0, pitch))
            pitch_s = float(inst['pitch'] == 'high') * (pitch_s - 236.0) / 42.0 + float(inst['pitch'] == 'low') * (278.0 - pitch_s) / 42.0
    else:
        pitch_s = FAILED_TOKEN
    return pitch, pitch_s


def calculate_speed(audio, transcript, inst='', language='english'):
    speed = ((len(transcript.split(' ')) if language == 'english' else len(transcript)) * SAMPLE_RATE * 60 / len(librosa.effects.trim(audio)[0]))
    if 'speed' in inst:
        if language == 'english':
            speed_s = max(156.0, min(180.0, speed))
            speed_s = float(inst['speed'] == 'high') * (speed_s - 156.0) / 24.0 + float(inst['speed'] == 'low') * (180.0 - speed_s) / 24.0
        else:
            speed_s = max(232.0, min(272.0, speed))
            speed_s = float(inst['speed'] == 'high') * (speed_s - 232.0) / 40.0 + float(inst['speed'] == 'low') * (272.0 - speed_s) / 40.0
    else:
        speed_s = FAILED_TOKEN
    return speed, speed_s


def calculate_volume(audio):
    audio = librosa.effects.trim(audio)[0]
    rms = librosa.feature.rms(y=audio)[0]
    return float(np.mean(rms))


def calculate_speech_similarity(audio_list, ref_audio_list, batch_size=8):
    if len(audio_list) == 0:
        return []
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').to('cuda')
    audio_list = [librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=16000) for audio in audio_list]
    ref_audio_list = [librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=16000) for audio in ref_audio_list]
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(audio_list), batch_size):
            inputs = feature_extractor(audio_list[i: min(i + batch_size, len(audio_list))],
                                       return_tensors="pt", padding=True, sampling_rate=16000).to('cuda')
            embeddings.append(model(**inputs).embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        ref_embeddings = []
        for i in range(0, len(ref_audio_list), batch_size):
            ref_inputs = feature_extractor(ref_audio_list[i: min(i + batch_size, len(ref_audio_list))],
                                           return_tensors="pt", padding=True, sampling_rate=16000).to('cuda')
            ref_embeddings.append(model(**ref_inputs).embeddings)
        ref_embeddings = torch.cat(ref_embeddings, dim=0)
        cos_sim = F.cosine_similarity(embeddings, ref_embeddings)
    return cos_sim.tolist()


def text_instruction_following_verify(text_list, instruction_list):
    """
    The instruction list should have two parameters, instruction_type and instruction_params.
    The return will be a list of {0, 1}s representing the instruction following result for each.
    """
    from transformers import AutoProcessor
    output_list = []
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    for text, inst in zip(text_list, instruction_list):
        text = text.lower().strip()
        if isinstance(inst[0], str):
            inst = [inst]
        flag = True
        for i in inst:
            if i[0] == 'exact_match':
                flag = text == processor.tokenizer._normalize(i[1]) and flag
            elif i[0] == 'keyword_include':
                flag = all(keyword in text for keyword in i[1:]) and flag
            elif i[0] == 'keyword_exclude':
                flag = not any(keyword in text for keyword in i[1:]) and flag
            elif i[0] == 'keyword_count':
                count = len(re.findall(re.escape(i[1]), text))
                flag = eval(f'count {i[2]}') and flag
            elif i[0] == 'length_word':
                words = word_tokenize(text)
                count = len([word for word in words if word.isalnum() or "'" in word])
                flag = eval(f'count {i[1]}') and flag
            elif i[0] == 'start':
                flag = text.startswith(processor.tokenizer._normalize(i[1])) and flag
            elif i[0] == 'end':
                flag = text.endswith(processor.tokenizer._normalize(i[1])) and flag
            else:
                raise NotImplementedError
        output_list.append(float(flag))
    return output_list


def extract_json(text):
    match = re.search(r'\{.*?}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            cleaned = re.sub(r'\s+', ' ', json_str).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {}
    return {}


def extract_list(text):
    match = re.search(r'\[.*?]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return []
