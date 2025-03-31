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
