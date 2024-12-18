import json
import os
import soundfile as sf
from abc import abstractmethod

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
    model_name = 'dalle3'

    def generate(self, query_list):
        image_list = batch(generate_image_from_openai, query_list, model="dall-e-3")
        res_list = []
        for query, image in zip(query_list, image_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0),
                'image_list': [image],
                'audio_list': []
            })
        return res_list

