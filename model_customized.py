from model import Model
from utils import IMAGE_TOKEN, AUDIO_TOKEN, SAMPLE_RATE


class ModelCustomized(Model):
    def __init__(self):
        """
        Anything needed to initialize the model.
        """
        pass

    def generate(self, query_list):
        """
        Args: query_list:
        Format of a query (dict) in the query list:
        {
            # instruction: instruction in text
            "instruction": "Can you give me a step-by-step tutorial of how to make tomato soup?",
            # image_list: Dirs (e.g. './seed_instruction/images/i_structure_0.png') of PNG images in the instruction, can be an empty list
            # You may use PIL.Image.Open(dir) to open the image or customized
            "image_list": [],
            # audio_list: Dirs (e.g. './seed_instruction/audio/a_structure_0.wav') od WAV audios in the instruction, can be an empty list
            # You may use librosa.load(dir, sr=SAMPLE_RATE) to open the audio or customized
            "audio_list": [],
        }

        Returns: res_list: List[dict]
        Format of a response (dict) in the result list:
        {
            # query (optional for debugging): should be the direct copy of input query
            "query": {"instruction": "Can you give me a step-by-step tutorial of how to make tomato soup?", "image_list": [], "audio_list": []},
            # response: Interleaved text, image and audio in the generated order; the response should be the generated text with images and audios replaced by placeholders
            # image placeholder: IMAGE_TOKEN(i) = <image_begin><image_i><image_end>, where i is the relative index of the image in the response starting from 0; e.g. IMAGE_TOKEN(0) is the first generated image.
            # audio placeholder: AUDIO_TOKEN(i) = <audio_begin><audio_i><audio_end>, where i is the relative index of the audio in the response starting from 0; e.g. AUDIO_TOKEN(0) is the first generated audio.
            # For image-only or audio-only generation models, you can return a single placeholder IMAGE_TOKEN(0) or AUDIO_TOKEN(0).
            "response": "Sure! Here is a step-by-step tutorial of how to make tomato soup: First, wash the tomatoes with clean water. <image_begin><image_0><image_end>. Second, ...",
            # image_list: images for corresponding image placeholders in order; images should be in PIL format
            "image_list": [PIL.Image, ...],
            # audio_list: audios for corresponding audio placeholders in order; audios should be in np.ndarray format with a sample rate of SAMPLE_RATE = 22050
            "audio_list": [np.ndarray, ...],
        }
        """
        pass
