import requests
import io

from model import Model
from utils import *


class Dalle3(Model):
    model_name = 'dall-e-3'

    def __init__(self, revise=True):
        self.revise_prompt = '' if revise else ('I NEED to test how the tool works with extremely simple prompts. '
                                                'DO NOT add any detail, just use it AS-IS:')

    @staticmethod
    def generate_image(index, prompt, model_name):
        import openai
        retry_count = 2
        retry_interval = 10

        if 'dall' in model_name:
            client = openai.OpenAI(api_key=OPENAI_KEY)
        elif 'recraft' in model_name:
            client = openai.OpenAI(base_url='https://external.api.recraft.ai/v1', api_key=RECRAFT_KEY)
        else:
            raise NotImplementedError

        for _ in range(retry_count):
            try:
                response = client.images.generate(
                    model=model_name,
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

    def generate(self, query_list):
        query_list = [self.revise_prompt + query['instruction'] for query in query_list]
        image_list = batch(self.generate_image, query_list, model_name=self.model_name)
        res_list = []
        for query, image in zip(query_list, image_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0) if image is not None else '',
                'image_list': [image] if image is not None else [],
                'audio_list': []
            })
        return res_list


class Recraft3(Dalle3):
    model_name = 'recraftv3'


class Imagen3(Model):
    model_name = 'imagen-3.0-generate-002'

    @staticmethod
    def generate_image_from_google(index, prompt, model_name):
        from google import genai
        from google.genai import types
        retry_count = 2
        retry_interval = 10
        client = genai.Client(api_key=GEMINI_KEY)

        for _ in range(retry_count):
            try:
                response = client.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio='1:1',
                    )
                )
                images = Image.open(BytesIO(response.generated_images[0].image.image_bytes))
                return index, images

            except Exception as e:
                print(f"Error info: {e}")
                print('Retrying....')
                retry_count += 1
                time.sleep(retry_interval * (2 ** retry_count))

        print('Failed to get response.')
        return index, None

    def generate(self, query_list):
        query_list = [query['instruction'] for query in query_list]
        image_list = batch(self.generate_image_from_google, query_list, model_name=self.model_name, num_worker=1)
        res_list = []
        for query, image in zip(query_list, image_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0) if image is not None else '',
                'image_list': [image] if image is not None else [],
                'audio_list': []
            })
        return res_list


class StableDiffusion3_5(Model):
    model_name = "stabilityai/stable-diffusion-3.5-large"

    def __init__(self):
        from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

        self.device = torch.device("cuda:0")
        transformer_model = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            token=HF_KEY,
            trust_remote_code=True
        )
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_name,
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
            token=HF_KEY,
            trust_remote_code=True
        )
        self.pipeline.enable_model_cpu_offload()

    def generate(self, query_list):
        prompts = [query['instruction'] for query in query_list]
        images = []
        batch_size = 64
        for i in range(0, len(prompts), batch_size):
            begin = i
            end = min(len(prompts), i + batch_size)
            images += self.pipeline(
                prompt=prompts[begin: end],
                num_inference_steps=28,
                height=512,
                width=512,
                guidance_scale=4.5,
            ).images

        res_list = []
        for image, query in zip(images, query_list):
            res_list.append({
                "query": query,
                "response": IMAGE_TOKEN(0),
                "image_list": [image],
                "audio_list": [],
            })
        return res_list


class ReplicateModel(Model):
    model_name: str
    image_size = "1024x1024"

    def __init__(self):
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_KEY

    @staticmethod
    def generate_image(index, query, model_name, image_size):
        import replicate
        retry_count = 2
        retry_interval = 10
        for _ in range(retry_count):
            try:
                input_params = {
                    "prompt": query.get("instruction", "")
                }
                if 'x' in image_size:
                    input_params['size'] = image_size
                else:
                    input_params['"aspect_ratio"'] = image_size
                output = replicate.run(
                    model_name,
                    input=input_params,
                )
                image_bytes = output.read()
                image = Image.open(BytesIO(image_bytes))
                return index, image

            except Exception as e:
                print(f"Error info: {e}")
                print('Retrying....')
                retry_count += 1
                time.sleep(retry_interval * (2 ** retry_count))

        print('Failed to get response.')
        return index, None

    def generate(self, query_list):
        images = batch(self.generate_image, query_list, image_size=self.image_size, model_name=self.model_name, num_worker=4)
        output_list = []
        for image, query in zip(images, query_list):
            output_list.append({
                "query": query,
                'response': IMAGE_TOKEN(0) if image is not None else '',
                'image_list': [image] if image is not None else [],
                "audio_list": [],
            })
        return output_list


class LumaPhoton(ReplicateModel):
    model_name = "luma/photon"


class Flux1_1Pro(ReplicateModel):
    model_name = "black-forest-labs/flux-1.1-pro"
    image_size = "512x512"


class Ideogram2(ReplicateModel):
    model_name = "ideogram-ai/ideogram-v2"
    image_size = "1:1"


class GPT4o(Model):
    model_name = "gpt-image-1"

    @staticmethod
    def generate_image(index, prompt, model_name):
        import openai
        retry_count = 2
        retry_interval = 10
        client = openai.OpenAI(api_key=OPENAI_KEY)

        for _ in range(retry_count):
            try:
                if 'image_list' not in prompt or len(prompt['image_list']) == 0:
                    response = client.images.generate(
                        model=model_name,
                        prompt=prompt['instruction'],
                        size="1024x1024",
                        quality="medium"
                    )
                else:
                    response = client.images.edit(
                        model=model_name,
                        image=[open(img, "rb") if isinstance(img, str) else img for img in prompt['image_list']],
                        prompt=prompt['instruction'],
                        size="1024x1024",
                        quality="medium"
                    )
                image_bytes = base64.b64decode(response.data[0].b64_json)
                return index, Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                print("Error info: ", e)
                print('Retrying....')
                retry_interval *= 2
                time.sleep(retry_interval)

        print('Fail to get response.')
        return index, None

    def generate(self, query_list):
        image_list = batch(self.generate_image, query_list, model_name=self.model_name)
        res_list = []
        for query, image in zip(query_list, image_list):
            res_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0) if image is not None else '',
                'image_list': [image] if image is not None else [],
                'audio_list': []
            })
        return res_list
