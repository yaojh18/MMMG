import random

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
        super().__init__()
        from models.tango.tango import Tango
        self.model = Tango("declare-lab/tango2")

    def generate(self, query_list):

        import os
        res_list = []
        random.seed(0)
        for query in query_list:

            ## generate audio (the generated audio is in integer type; not float)
            ## save it & reload it changes it to float type
            audio=self.model.generate(query['instruction'])#, seed=random.randint(0, 1000)).numpy(),
            sf.write("audio.wav", audio, samplerate=16000)            
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [librosa.resample(
                    librosa.load("audio.wav", sr=16000)[0],
                    orig_sr=16000,
                    target_sr=SAMPLE_RATE
                )]
            })
        return res_list


class StableAudio(Model):
    def __init__(self):
        super().__init__()
        from huggingface_hub import login
        from diffusers import StableAudioPipeline        
        login(token=HF_KEY)
        self.model=StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
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

        import os
        import shutil
        os.chdir("./models/YuE/inference")
        with open("lyrics.txt", "w") as f:
            f.write("[verse]\n\n[chorus]\n\n[outro]")
            
    def generate(self, query_list):
        output_list = []        
        query_list = [query['instruction'] for query in query_list]
        
        for query in tqdm(query_list):
            with open("query.txt", "w") as f:
                f.write(query)
            command = f"""python infer.py --cuda_idx 0 \
                                        --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
                                        --stage2_model m-a-p/YuE-s2-1B-general \
                                        --genre_txt query.txt \
                                        --lyrics_txt lyrics.txt \
                                        --run_n_segments 2 \
                                        --stage2_batch_size 4 \
                                        --output_dir output/ \
                                        --max_new_tokens 500 \
                                        --repetition_penalty 1.1"""
            os.system(command)

            ## process output
            file = [item for item in os.listdir("output/") if '-'.join(query.split()) in item][0]            
            output_list.append(librosa.load(f"output/{file}")[0])
            shutil.rmtree("output/")
            os.makedirs("output/")
            
        res_list = []
        for query, output in zip(query_list, output_list):
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [output],
            })
        return res_list
