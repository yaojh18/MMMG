import os
import json
import random
import shutil

import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageDraw

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


def sample_from_isg_bench():
    data_list = []
    with open(f'./datasets/ISG-Bench/ISG-Bench.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['Category'] == 'multi-Perspective Scene Generation' and len(data['Golden']) >= 8 and len(data['Query']) == 2:
                data_list.append(json.loads(line.strip()))
    json_list = []
    idx = 0
    for data in data_list[:20]:
        order = [data['Golden'][1]['content'], data['Golden'][3]['content'], data['Golden'][5]['content'], data['Golden'][7]['content']]
        idxs = random.sample(list(range(4)), 4)
        json_list.append({
            'instruction': 'The given image represents the frontal observation scene, based on this, generate four additional images showing views from the following perspectives in order: ' + ', '.join([order[i] for i in idxs]) + '.',
            'image_list': [idx],
            'ref_image_list': list(range(idx + 1, idx + 5)),
        })
        image = Image.open('./datasets/ISG-Bench/' + data['Query'][1]['content'])
        image.save(f'./seed_instruction/image/i_consistency_3d_scene_{idx}.png')
        for i, j in enumerate(idxs):
            image = Image.open('./datasets/ISG-Bench/' + data['Golden'][j * 2]['content'])
            image.save(f'./seed_instruction/image/i_consistency_3d_scene_{idx + i + 1}.png')
        idx += 5
    with open('seed_instruction/i_consistency_3d_scene.jsonl', 'w', encoding='utf-8') as file:
        for data in json_list:
            file.write(json.dumps(data) + '\n')


def sample_from_openmic():
    df = pd.read_csv('./datasets/temp/openmic-2018-v1.0.0/openmic-2018/openmic-2018-aggregated-labels.csv')
    df = df[df['relevance'] == 1]
    id_counts = df['sample_key'].value_counts()
    unique_instruments = id_counts[id_counts == 1].index
    df = df[df['sample_key'].isin(unique_instruments)]
    instrument_list = ['banjo', 'bass', 'cello', 'clarinet', 'cymbals', 'mandolin', 'trombone', 'trumpet', 'ukulele', 'violin']
    df = df[df['instrument'].isin(instrument_list)]
    df = df.groupby('instrument')
    for instrument_name, group_data in df:
        os.makedirs(f'./datasets/openmic-2018/{instrument_name}/', exist_ok=True)
        for idx, (_, row) in enumerate(group_data.iterrows()):
            shutil.copy(f"./datasets/temp/openmic-2018-v1.0.0/openmic-2018/mp3/audio/{row['sample_key']}.mp3",
                        f'./datasets/openmic-2018/{instrument_name}/{idx}.mp3')


if __name__ == '__main__':
    pass