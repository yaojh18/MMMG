import random
import os
import shutil
import string

from model import Model
from utils import *


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


class Tango2(Model):
    def __init__(self):
        from models.tango.tango import Tango
        self.model = Tango("declare-lab/tango2-full")

    def generate(self, query_list):
        res_list = []
        random.seed(0)
        for query in tqdm(query_list):
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [librosa.resample(
                    self.model.generate(query['instruction'], steps=200).astype(np.float32)/np.iinfo(np.int16).max,
                    orig_sr=16000,
                    target_sr=SAMPLE_RATE
                )]
            })
        return res_list


class TangoMusic(Tango2):
    def __init__(self):
        from models.tango.tango import Tango
        self.model = Tango("declare-lab/tango-music-af-ft-mc")


class StableAudio(Model):
    def __init__(self):
        super().__init__()
        from huggingface_hub import login
        from diffusers import StableAudioPipeline        
        login(token=HF_KEY)
        self.model = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
        self.model = self.model.to("cuda")
        
    def generate(self, query_list):
        res_list = []
        random.seed(0)
        for query in query_list:
            audio = self.model(
                query['instruction'],
                num_inference_steps=200,
                audio_end_in_s=10.0,
                num_waveforms_per_prompt=3,
                generator=torch.Generator("cuda").manual_seed(random.randint(0, 1000))
            ).audios
            
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [librosa.resample(
                    librosa.to_mono(audio[0].float().cpu().numpy()),
                    orig_sr=self.model.vae.sampling_rate,
                    target_sr=SAMPLE_RATE
                )]
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


class YuE(Model):
    def __init__(self):
        from model import GeminiModel

        system_prompt = """Rephrase the given sentence to match the style of the example provided. 

        Input 1: Create a guitar solo that conveys strong emotion and expression.        
        Output 1: emotional expressive solo guitar

        Input 2: Create a majestic and grand classical flute composition with a tempo of 145 BPM.        
        Output 2: majestic classical flute 145BPM

        Input:"""

        self.genre_corrector = GeminiModel('gemini-2.0-flash', system_prompt)
        os.chdir("./models/YuE/inference")
        with open("lyrics.txt", "w") as f:
            f.write("[verse]\n\n[chorus]\n\n[outro]")

    def generate(self, query_list):
        ## convert instructino to YuE compatible format
        query_list = self.genre_corrector.generate(query_list)
        query_list = [q.replace('### Assistant:', '').strip().lower().translate(str.maketrans('', '', string.punctuation))
                      for q in query_list]
        output_list = []

        for query in tqdm(query_list):
            with open("models/YuE/prompt_egs/genre.txt", "w") as f:
                f.write(query)
            command = f"""python models/YuE/inference/infer.py --cuda_idx 2 \
                                                               --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
                                                               --stage2_model m-a-p/YuE-s2-1B-general \
                                                               --genre_txt models/YuE/prompt_egs/genre.txt \
                                                               --lyrics_txt models/YuE/prompt_egs/lyrics.txt \
                                                               --run_n_segments 2 \
                                                               --stage2_batch_size 4 \
                                                               --output_dir models/YuE/output/ \
                                                               --max_new_tokens 500 \
                                                               --repetition_penalty 1.1"""
            os.system(command)

            ## process output
            file = [item for item in os.listdir("models/YuE/output/") if item.endswith('.mp3')][0]
            output_list.append(librosa.load(f"models/YuE/output/{file}")[0])
            shutil.rmtree("models/YuE/output/")
            os.makedirs("models/YuE/output/")

        res_list = []
        for query, output in zip(query_list, output_list):
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [output],
            })
        return res_list


class AudioGen(Model):
    def __init__(self):
        from audiocraft.models import AudioGen
        self.model = AudioGen.get_pretrained("facebook/audiogen-medium")

    def generate(self, query_list):
        from audiocraft.data.audio import audio_write
        res_list = []
        for query in tqdm(query_list):
            audio = self.model.generate([query['instruction']])
            audio_write('./output/AudioGen/temp', audio.squeeze(0).cpu(), self.model.sample_rate)
            audio, _ = librosa.load('./output/AudioGen/temp.wav', sr=SAMPLE_RATE)
            res_list.append({
                "query": query,
                "response": AUDIO_TOKEN(0),
                "image_list": [],
                "audio_list": [audio],
            })
        os.remove('./output/AudioGen/temp.wav')
        return res_list


class Magnet(AudioGen):
    def __init__(self):
        from audiocraft.models import MAGNeT
        self.model = MAGNeT.get_pretrained("facebook/magnet-medium-10secs")


class AudioLDM2(Model):
    def __init__(self):
        from diffusers import AudioLDM2Pipeline
        self.pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def generate(self, query_list):
        res_list = []
        random.seed(0)
        for query in query_list:
            audio = self.pipe(
                query['instruction'],
                num_inference_steps=200,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=3,
                generator=torch.Generator("cuda").manual_seed(random.randint(0, 1000))
            ).audios

            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [librosa.resample(librosa.to_mono(audio), orig_sr=16000, target_sr=SAMPLE_RATE)]
            })
        return res_list


class MakeAnAudio2(Model):
    def generate(self, query_list):
        query_list = [query['instruction'] + '\n' for query in query_list]
        os.makedirs('./models/Make-An-Audio-2/input', exist_ok=True)
        with open('./models/Make-An-Audio-2/input/prompts.txt', 'w', encoding='utf-8') as f:
            f.writelines(query_list)
        os.chdir("./models/Make-An-Audio-2")
        if os.path.exists('./output'):
            shutil.rmtree('./output')
            os.makedirs('./output')
        os.system("PYTHONPATH=. python scripts/gen_wav.py")
        os.chdir("../..")
        res_list = []
        for idx, query in enumerate(query_list):
            audio, sr = librosa.load(f'./models/Make-An-Audio-2/output/{idx}.wav', sr=SAMPLE_RATE)
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [audio],
            })
        return res_list
