import random
from abc import abstractmethod

import torch

from utils import *


class Model:
    model_name: str

    @abstractmethod
    def generate(self, query_list):
        """
        :param sample_size:
        :param save_name: str
        :param query_list: List[str]
        :return: res_list
        Format of an item in the response list:
        {
            "query": "Can you give me a step-by-step tutorial of how to make tomato soup?",
            "response": "Sure! Here is a step-by-step tutorial of how to make tomato soup: First, wash the tomatoes with clean water. <image_begin><image_1><image_end>. Second, ..."
            "image_list": [PIL.image, ...],
            "audio_list": [np.ndarray, ...]
        }
        """
        pass


class Dalle3(Model):
    model_name = 'dall-e-3'

    def generate(self, query_list):
        image_list = batch(generate_image_from_openai, query_list, model=self.model_name)
        res_list = []
        for query, image in zip(query_list, image_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0),
                'image_list': [image],
                'audio_list': []
            })
        return res_list


class OmniGen(Model):
    model_name = 'omnigen'

    def __init__(self):
        super().__init__()
        from OmniGen import OmniGenPipeline
        self.batch_size = 16
        self.sample_size = 4
        self.pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

    def generate(self, query_list):
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


class TangoFlux(Model):
    model_name = 'tango-flux'

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
                'audio_list': [self.model.generate(query, steps=50, duration=5, seed=random.randint(0, 1000))],
            })
        return res_list
