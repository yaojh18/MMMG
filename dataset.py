import os
import json
import random
from datasets import load_dataset
from PIL import Image, ImageDraw
import gradio as gr
from gradio_image_prompter import ImagePrompter
import matplotlib.pyplot as plt

from interface import LabelBBoxInterface


def curate_image_editing_instruction():
    raw_dataset = load_dataset('facebook/emu_edit_test_set')['test']
    for task in ('text', 'add', 'remove', 'local'):
        dataset = []
        for data in raw_dataset:
            if data['task'] == task:
                dataset.append({
                    'instruction': data['instruction'],
                    'image_list': [data['image']],
                    'input_caption': data['input_caption'],
                    'output_caption': data['output_caption'],
                })
        dataset = random.sample(dataset, 60)
        output_path = f'./data/emuedit/'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path + 'image/', exist_ok=True)
        image_list = []
        output_list = []
        image_idx = 0
        for res in dataset:
            output = res.copy()
            output['image_list'] = list(range(image_idx, image_idx + len(res['image_list'])))
            image_idx += len(res['image_list'])
            image_list += res['image_list']
            output_list.append(output)
        with open(output_path + f'{task}.jsonl', 'w', encoding='utf-8') as file:
            for data in output_list:
                file.write(json.dumps(data) + '\n')
        for idx, image in enumerate(image_list):
            image.save(output_path + f'image/{task}_{idx}.png')


def label_image_editing_instruction():
    for task in ('local',):
        dataset = []
        with open(f'./data/emuedit/{task}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        for data in dataset:
            data['image'] = f'./data/emuedit/image/{task}_{data["image_list"][0]}.png'
        interface = LabelBBoxInterface(data_list=dataset)
        interface.start()
        output_list = []
        for data, bbox in zip(dataset, interface.res_list):
            if bbox is not None:
                bbox = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
                data['bbox'] = bbox
                del data['image']
                output_list.append(data)
        with open(f'./data/emuedit/{task}.jsonl', 'w', encoding='utf-8') as file:
            for data in output_list:
                file.write(json.dumps(data) + '\n')


def validate_image_editing_instruction():
    for task in ('local',):
        dataset = []
        with open(f'./data/emuedit/{task}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        for data in dataset:
            image = Image.open(f'./data/emuedit/image/{task}_{data["image_list"][0]}.png')
            draw = ImageDraw.Draw(image)
            draw.rectangle(data['bbox'], outline="red", width=3)
            plt.imshow(image)
            plt.axis('off')
            plt.show()


def reindex_image_editing_instruction():
    name_translator = {
        'text': 'text',
        'add': 'object_add',
        'remove': 'object_remove',
        'local': 'object_modify',
    }
    for task in ('remove', 'local',):
        dataset = []
        with open(f'./data/emuedit/{task}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        for data in dataset:
            data['image_list'] = [Image.open(f'./data/emuedit/image/{task}_{idx}.png') for idx in data['image_list']]
        output_path = f'./seed_instruction/'
        image_list = []
        output_list = []
        image_idx = 0
        for data in dataset:
            output = data.copy()
            output['image_list'] = list(range(image_idx, image_idx + len(data['image_list'])))
            image_idx += len(data['image_list'])
            image_list += data['image_list']
            output_list.append(output)
        with open(output_path + f'i_edit_{name_translator[task]}.jsonl', 'w', encoding='utf-8') as file:
            for data in output_list:
                file.write(json.dumps(data) + '\n')
        for idx, image in enumerate(image_list):
            image.save(output_path + f'image/i_edit_{name_translator[task]}_{idx}.png')


if __name__ == '__main__':
    demo = gr.Interface(
        lambda prompts: (prompts["image"], prompts["points"]),
        ImagePrompter(show_label=False),
        [gr.Image(show_label=False), gr.Dataframe(label="Points")],
    )
    demo.launch()
