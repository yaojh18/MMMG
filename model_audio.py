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
        from tango import Tango
        self.model = Tango("declare-lab/tango2-full")

    def generate(self, query_list):
        res_list = []
        random.seed(0)
        for query in query_list:
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [librosa.resample(
                    self.model.generate(query['instruction'], seed=random.randint(0, 1000)).numpy(),
                    orig_sr=16000,
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
            
    def generate(self, query_list):
        output_list = []
        lyrics = "[verse]\n\n[chorus]\n\n[outro]"
        query_list = [query['instruction'] for query in query_list]

        for query in tqdm(query_list):

            ## generate music in cmd
            os.system(f"""python infer.py \
                        --cuda_idx 0 \
                        --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
                        --stage2_model m-a-p/YuE-s2-1B-general \
                        --genre_txt {query} \
                        --lyrics_txt {lyrics} \
                        --run_n_segments 2 \
                        --stage2_batch_size 4 \
                        --output_dir ../output \
                        --max_new_tokens 500 \
                        --repetition_penalty 1.1
                        """)
            
            file = [i for i in os.listdir('../output/') if '-'.join(query.split()) in i][0]            
            output_list.append(librosa.load('file')[0])    
            shutil.rmtree('../output/')
            os.makedirs('../output/')
            
        res_list = []
        for query, output in zip(query_list, output_list):
            res_list.append({
                'query': query,
                'response': AUDIO_TOKEN(0),
                'image_list': [],
                'audio_list': [output],
            })
        return res_list