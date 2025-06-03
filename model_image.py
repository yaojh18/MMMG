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


class Janus(Model):
    def __init__(self):
        from transformers import AutoModelForCausalLM
        from models.Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
        model_path = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def generate_image(
            self,
            prompt: str,
            temperature: float = 0.1,
            parallel_size: int = 1,
            cfg_weight: float = 5,
            image_token_num_per_image: int = 576,
            img_size: int = 384,
            patch_size: int = 16,
    ):
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = self.vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                                             past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        return Image.fromarray(visual_img[0])

    def generate(self, query_list):
        output_list = []
        for query in tqdm(query_list):
            conversation = [
                {
                    "role": "<|User|>",
                    "content": query['instruction'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag
            output_list.append({
                'query': query,
                'response': IMAGE_TOKEN(0),
                'image_list': [self.generate_image(prompt)],
                'audio_list': []
            })
        return output_list


class Bagel(Model):
    def __init__(self):
        from models.BAGEL.modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel
        )
        from models.BAGEL.modeling.autoencoder import load_ae
        from models.BAGEL.modeling.qwen2 import Qwen2Tokenizer
        from models.BAGEL.data.data_utils import add_special_tokens
        from models.BAGEL.data.transforms import ImageTransform
        from models.BAGEL.inferencer import InterleaveInferencer
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

        torch.cuda.empty_cache()
        model_path = './models/BAGEL/BAGEL-7B-MoT'
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

    def generate(self, query_list):
        from models.BAGEL.inferencer import set_seed
        set_seed(0)
        output_list = []
        inference_hyper = dict(
            max_think_token_n=1024,
            do_sample=True,
            text_temperature=0.1,
            cfg_text_scale=4.0,
            cfg_interval=[0.4, 1.0],  # End fixed at 1.0
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(1024, 1024),
            think=True,
        )
        for query in tqdm(query_list):
            reformed_query = {'text': query['instruction']}
            if 'image_list' in reformed_query:
                reformed_query['image'] = query['image_list']
            output = self.inferencer(**reformed_query, **inference_hyper)
            response = ''
            image_list = []
            idx = 0
            for out in output:
                if isinstance(out, Image.Image):
                    response += IMAGE_TOKEN(idx)
                    idx += 1
                    image_list.append(out)
                else:
                    response += out
            output_list.append({
                'query': query,
                'response': response,
                'image_list': image_list,
                'audio_list': []
            })
        return output_list

