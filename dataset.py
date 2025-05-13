import cairosvg
import os
import random
import shutil
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import ImageDraw

from interface import LabelBBoxInterface
from utils import *


def sample_from_emu_edit():
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
    for task in ('i_edit_object_add', ):
        dataset = []
        with open(f'./seed_instruction/{task}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        for data in dataset:
            data['image'] = f'./seed_instruction/image/{task}_{data["image_list"][0]}.png'
        interface = LabelBBoxInterface(data_list=dataset)
        interface.start()
        output_list = []
        for data, bbox in zip(dataset, interface.res_list):
            if bbox is not None:
                bbox = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
                data['bbox'] = bbox
                del data['image']
                output_list.append(data)
        with open(f'./seed_instruction/{task}.jsonl', 'w', encoding='utf-8') as file:
            for data in output_list:
                file.write(json.dumps(data) + '\n')


def validate_image_editing_instruction():
    for task in ('i_edit_object_add', ):
        dataset = []
        with open(f'./seed_instruction/{task}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        for data in dataset:
            image = Image.open(f'./seed_instruction/image/{task}_{data["image_list"][0]}.png')
            draw = ImageDraw.Draw(image)
            draw.rectangle(data['bbox'], outline="red", width=3)
            plt.imshow(image)
            plt.axis('off')
            plt.show()


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
    instrument_list = ['piano']
    df = df[df['instrument'].isin(instrument_list)]
    df = df.groupby('instrument')
    for instrument_name, group_data in df:
        os.makedirs(f'./datasets/openmic-2018/{instrument_name}/', exist_ok=True)
        for idx, (_, row) in enumerate(group_data.iterrows()):
            shutil.copy(f"./datasets/temp/openmic-2018-v1.0.0/openmic-2018/mp3/audio/{row['sample_key']}.mp3",
                        f'./datasets/openmic-2018/{instrument_name}/{idx}.mp3')


def paraphrasing_dataset(file_name):
    res_list = []
    with open(f'./seed_instruction/{file_name}.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            res_list.append(json.loads(line.strip()))
    instruction = lambda i: f"### Instruction:\n Polish the following instruction to improve clarity and readability while preserving 100% of the original semantic meaning. Do not add, delete, or modify any requirements or specifications from the original instruction. Make small change. \n ### Instruction:\n{i}\n### Output format:\n ONLY the instruction after paraphrasing."
    queries = [form_openai_mm_query(instruction(res['instruction'])) for res in res_list]
    responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.5)
    for data, res in zip(res_list, responses):
        data['instruction_para'] = res.strip()
    with open(f'./seed_instruction/{file_name}.jsonl', 'w', encoding='utf-8') as file:
        for res in res_list:
            file.write(json.dumps(res) + '\n')


def sample_from_star_vector():
    dataset = load_dataset('starvector/svg-emoji')
    collected_data = []
    for data in dataset['test']:
        if 1000 < len(data['Svg']) < 2000:
            collected_data.append(data['Svg'])
    random.shuffle(collected_data)
    with open(f'./seed_instruction/it_coherence_code.jsonl', 'w', encoding='utf-8') as f:
        for i, data in enumerate(collected_data[:100]):
            f.write(json.dumps({f'instruction': f"### SVG Code:\n{data}\n### Instruction:\nWhat does this SVG code represent? Analyze the elements step by step, then create a rendered image showing how it would appear in a browser. \n", 'ref_image_list': [i], 'instruction_para': f"### SVG Code:\n{data}\n### Instruction:\nWhat does this SVG code represent? Analyze the elements step by step, then create a rendered image showing how it would appear in a browser.\n"}) + '\n')
    for i, data in enumerate(collected_data[:100]):
        cairosvg.svg2png(bytestring=data, write_to=f'./seed_instruction/image/it_coherence_code_{i}.png', output_width=1024, output_height=1024)


def create_huggingface_dataset():
    from huggingface_hub import login, create_repo, upload_folder
    import pyarrow as pa
    import pyarrow.parquet as pq
    import glob
    repo_id = 'UW-FMRL2/MMMG'
    login(token=HF_KEY)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False
        )
    except:
        pass
    os.makedirs('./hf_dataset/test/', exist_ok=True)
    os.makedirs('./hf_dataset/test/media/', exist_ok=True)

    output_list = []
    jsonl_list = glob.glob('./seed_instruction/*.jsonl')
    for jsonl in jsonl_list:
        task_name = re.search(r'\./seed_instruction/(.*).jsonl', jsonl).group(1)
        with open(jsonl, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                new_data = {
                    'category': task_name,
                    'seed_instruction': data['instruction'],
                    'instruction': data['instruction_para'],
                    'image_0': None, 'image_1': None, 'ref_image_0': None, 'ref_image_1': None,
                    'ref_image_2': None, 'ref_image_3': None, 'audio_0': None, 'ref_audio_0': None,
                }
                if 'image_list' in data:
                    for idx, i in enumerate(data['image_list']):
                        new_data[f'image_{idx}'] = open(f'./seed_instruction/image/{task_name}_{i}.png', 'rb').read()
                if 'audio_list' in data:
                    for idx, i in enumerate(data['audio_list']):
                        new_data[f'audio_{idx}'] = open(f'./seed_instruction/audio/{task_name}_{i}.wav', 'rb').read()
                if 'ref_image_list' in data:
                    for idx, i in enumerate(data['ref_image_list']):
                        new_data[f'ref_image_{idx}'] = open(f'./seed_instruction/image/{task_name}_{i}.png', 'rb').read()
                if 'ref_audio_list' in data:
                    for idx, i in enumerate(data['ref_image_list']):
                        new_data[f'ref_audio_{idx}'] = open(f'./seed_instruction/audio/{task_name}_{i}.wav', 'rb').read()
                output_list.append(new_data)
    output_df = pd.DataFrame(output_list)
    table = pa.Table.from_pandas(output_df)
    pq.write_table(table, './hf_dataset/test.parquet')

    upload_folder(
        folder_path="./hf_dataset/",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload initial dataset files",
    )


if __name__ == '__main__':
    create_huggingface_dataset()
