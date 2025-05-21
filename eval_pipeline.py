from eval_image import *
from eval_audio import *
from eval_interleaved import *
from utils import *

import pandas as pd
import argparse
from collections import defaultdict


class EvalPipeline:
    def __init__(self, model_name, cat='i', sample_size=1):
        assert cat in ['i', 'it', 'a', 'at']
        self.model_name = model_name
        self.cat = cat
        self.sample_size = sample_size
        if cat == 'i':
            self.eval_list = ['object inclusion', 'object exclusion', 'object count', 'object reasoning',
                              'object attribution', 'comparison relation', 'universal relation', 'relative spatial relation',
                              'absolute spatial relation', 'region fill', 'border fill', 'single text rendering',
                              'double text rendering', 'multi-lingual text rendering']
        elif cat == 'it':
            self.eval_list = ['semantic consistency', 'multi-angle consistency', 'multi-view consistency', 'composition consistency',
                              'decomposition consistency', 'interleaved object adding', 'interleaved color modifying', 'text editing',
                              'object adding', 'object removing', 'object_modifying', 'self count',
                              'self color recognition', 'self size recognition', 'self text recognition', 'self relative spatial recognition',
                              'self absolute spatial recognition', 'interleaved math', 'interleaved code', 'text-image order']
        elif cat == 'a':
            self.eval_list = ['sound begin-end', 'sound inclusion', 'sound reasoning', 'sound silence',
                              'instrument inclusion', 'instrument exclusion', 'music tempo', 'music intensity']
        elif cat == 'at':
            self.eval_list = ['voice attribution', 'multi-lingual speech', 'voice replication', 'transcript editing',
                              'transcript generation', 'conversation', 'audio-text order']

        if os.path.exists(f'./output/{model_name}/{cat}_eval.csv'):
            self.eval_df = pd.read_csv(f'./output/{model_name}/{cat}_eval.csv')
        else:
            self.eval_df = pd.DataFrame({
                'task': self.eval_list,
                'accuracy': [None] * len(self.eval_list),
                'agreement': [None] * len(self.eval_list),
                'correlation': [None] * len(self.eval_list)
            })
            os.makedirs(f'./output/{model_name}/', exist_ok=True)
            self.eval_df.to_csv(f'./output/{model_name}/{cat}_eval.csv', index=False)

    @staticmethod
    def eval_map(model_name, task_name, sample_size):
        class m_defaultdict(defaultdict):
            def __missing__(self, key):
                return key
        task_name = m_defaultdict(str, {
            'object inclusion': 'i_object_include', 'object exclusion': 'i_object_exclude',
            'object count': 'i_object_count', 'object reasoning': 'i_object_cot',
            'object attribution': 'i_object_attribute', 'comparison relation': 'i_relation_two',
            'universal relation': 'i_relation_all', 'relative spatial relation': 'i_spacial_relative',
            'absolute spatial relation': 'i_spacial_absolute', 'region fill': 'i_format_background',
            'border fill': 'i_format_border', 'single text rendering': 'i_ocr',
            'double text rendering': 'i_ocr_two', 'multi-lingual text rendering': 'i_ocr_multi_lingual',
            'semantic consistency': 'i_consistency_semantic', 'multi-angle consistency': 'i_consistency_3d_object',
            'multi-view consistency': 'i_consistency_3d_scene', 'composition consistency': 'i_consistency_compose',
            'decomposition consistency':'i_consistency_decompose', 'interleaved object adding':'i_edit_add',
            'interleaved color modifying': 'i_edit_color', 'text editing': 'i_edit_text',
            'object adding': 'i_edit_object_add', 'object removing': 'i_edit_object_remove',
            'object_modifying': 'i_edit_object_modify', 'self count': 'it_coherence_count',
            'self color recognition': 'it_coherence_color', 'self size recognition': 'it_coherence_size',
            'self text recognition': 'it_coherence_ocr', 'self relative spatial recognition': 'it_coherence_spacial_relative',
            'self absolute spatial recognition': 'it_coherence_spacial_absolute', 'interleaved math': 'it_coherence_math',
            'interleaved code': 'it_coherence_code', 'text-image order': 'i_structure',
            'sound begin-end': 'a_sound_begin_end', 'sound inclusion': 'a_sound_include',
            'sound reasoning': 'a_sound_cot', 'sound silence': 'a_sound_silence',
            'instrument inclusion': 'a_music_instrument', 'instrument exclusion': 'a_music_exclude',
            'music tempo': 'a_music_tempo', 'music intensity': 'a_music_intensity',
            'voice attribution': 'a_speech_attribute', 'multi-lingual speech': 'a_speech_chinese',
            'voice replication': 'a_speech_imitate', 'transcript editing': 'a_speech_modify',
            'transcript generation': 'a_speech_constraint', 'conversation': 'a_consistency_conversation',
            'audio-text order': 'a_structure'
        })[task_name]
        task_name = ''.join([t.capitalize() if i > 0 else t.upper() for i, t in enumerate(task_name.split('_'))])
        task_name = task_name.replace('Ocr', 'OCR').replace('Cot', 'CoT').replace('3d', '3D')
        return eval(f"{task_name}(model_name='{model_name}', sample_size={sample_size})")

    def generate(self):
        for index, row in self.eval_df.iterrows():
            task_name = row['task']
            self.eval_map(self.model_name, task_name, self.sample_size)

    def evaluate(self):
        for index, row in self.eval_df.iterrows():
            task_name = row['task']
            task = self.eval_map(self.model_name, task_name, self.sample_size)
            if pd.isna(row['accuracy']):
                task.evaluate()
            self.eval_df.loc[index, 'accuracy'] = task.compute_accuracy()
            self.eval_df.to_csv(f'./output/{self.model_name}/{self.cat}_eval.csv', index=False)

    def human_evaluate(self):
        for index, row in self.eval_df.iterrows():
            task_name = row['task']
            task = self.eval_map(self.model_name, task_name, self.sample_size)
            if pd.isna(row['accuracy']):
                task.evaluate()
            if pd.isna(row['agreement']):
                task.human_evaluate()
            self.eval_df.loc[index, 'accuracy'] = task.compute_accuracy()
            self.eval_df.loc[index, ['agreement', 'correlation']] = task.compute_correlation()
            self.eval_df.to_csv(f'./output/{self.model_name}/{self.cat}_eval.csv', index=False)

    def compute_ci(self):
        self.eval_df['ci'] = [None] * len(self.eval_df)
        all_score = []
        for index, row in self.eval_df.iterrows():
            task_name = row['task']
            task = self.eval_map(self.model_name, task_name, self.sample_size)
            eval_list = task.compute_accuracy(return_list=True)
            eval_list = np.array(eval_list).reshape(len(eval_list) // self.sample_size, self.sample_size)
            eval_score = np.mean(eval_list, axis=0)
            var = np.std(eval_score, ddof=1) / np.sqrt(self.sample_size) * 1.96
            self.eval_df.loc[index, 'ci'] = var
            all_score.append(eval_score)
        all_score = np.mean(all_score, axis=0)
        var = np.std(all_score, ddof=1) / np.sqrt(self.sample_size) * 1.96
        self.eval_df.to_csv(f'./output/{self.model_name}/{self.cat}_eval.csv', index=False)
        return var


class EvalBenchmark:
    def __init__(self, model_list=[], cat='i', sample_size=4):
        assert cat in ['i', 'it', 'a', 'at']
        self.cat = cat
        self.sample_size = sample_size
        self.pipelines = {}

        if cat == 'i':
            base_model_list = ['Imagen3', 'Recraft3', 'LumaPhoton', 'Flux1_1Pro', 'Ideogram2', 'Dalle3',
                               'StableDiffusion3_5', 'SeedLlama', 'Anole', 'Gemini2', 'GPT4o']
        elif cat == 'it':
            base_model_list = ['SeedLlama', 'Anole', 'Gemini2', 'GeminiAgent', 'GPT4oAgent', 'HybridAgent']
        elif cat == 'a':
            base_model_list = ['StableAudio', 'AudioLDM2', 'AudioGen', 'MakeAnAudio2', 'Tango2',
                               'MusicGen', 'TangoMusic', 'YuE']
        else:
            base_model_list = ['SpiritLM', 'VoxInstructAgent', 'VoiceLDMAgent']

        self.model_list = base_model_list + model_list

    def rank_models(self, method='absolute'):
        # ci_map = {}
        for model_name in self.model_list:
            pipeline = EvalPipeline(model_name, self.cat, self.sample_size)
            pipeline.evaluate()
            # ci_map[model_name] = pipeline.compute_ci()
            self.pipelines[model_name] = pipeline
        reshaped_dfs = []
        for model_name, pipeline in self.pipelines.items():
            temp_df = pipeline.eval_df[['task', 'accuracy']].copy()
            # temp_df.loc[len(temp_df)] = ['ci', ci_map[model_name]]
            temp_df.rename(columns={'accuracy': model_name}, inplace=True)
            reshaped_dfs.append(temp_df)

        combined_df = reshaped_dfs[0]
        for i in range(1, len(reshaped_dfs)):
            combined_df = pd.merge(combined_df, reshaped_dfs[i], on='task', how='inner')
        model_names = self.pipelines.keys()
        results = {'models': model_names}
        combined_df.to_csv(f'./output/{self.cat}_eval.csv', index=False)
        if method == 'absolute':
            scores = np.array(combined_df[model_names])
            scores = np.mean(scores, axis=0)
            results['scores'] = scores
            results['rank'] = self.rank_with_ties(scores)
        elif method == 'relative':
            scores_matrix = combined_df[model_names].values
            min_vals = np.min(scores_matrix, axis=1, keepdims=True)
            max_vals = np.max(scores_matrix, axis=1, keepdims=True)
            normalized_rows = np.zeros_like(scores_matrix)
            normalized_rows[:] = scores_matrix
            normalized_rows = (scores_matrix - min_vals) / (max_vals - min_vals)
            scores = normalized_rows.mean(axis=0).tolist()
            results['scores'] = scores
            results['rank'] = self.rank_with_ties(scores)
        elif method == 'rank':
            scores_matrix = combined_df[model_names].values
            rank_matrix = np.zeros_like(scores_matrix)
            for i in range(scores_matrix.shape[0]):
                rank_matrix[i, :] = self.rank_with_ties(-scores_matrix[i, :])
            scores = rank_matrix.mean(axis=0).tolist()
            results['scores'] = scores
            results['rank'] = self.rank_with_ties(scores)
        else:
            raise NotImplementedError

        results = pd.DataFrame(results)
        results.to_csv(f'./output/{self.cat}_avg.csv', index=False)
        return results

    @staticmethod
    def rank_with_ties(values, ascending=False):
        s = pd.Series(values)
        ranks = s.rank(method='min', ascending=ascending).astype(int).tolist()
        return ranks

    def compute_correlation(self):
        if self.cat == 'i':
            golden_reference = {'Imagen3': 1091, 'Recraft3': 1012, 'LumaPhoton': 1023, 'Flux1_1Pro': 1001,
                                'Ideogram2': 1022, 'Dalle3': 978, 'StableDiffusion3_5': 922}
            res = self.rank_models(method='absolute')
            print(calculate_pearson([golden_reference[m] for m in res['models'].to_list()], res['scores'].to_list()))
        else:
            print('There is no baseline for audio generation evaluation.')

    def human_evaluate(self, index=0):
        random_pipeline = EvalPipeline(model_name=f'RandomModel_{index}', cat=self.cat, sample_size=2)
        random_pipeline.human_evaluate()

    def merge_human_evaluation(self):
        merged_res = EvalPipeline(model_name='RandomModel', cat=self.cat, sample_size=2)
        for task_name in merged_res.eval_list:
            task = merged_res.eval_map('RandomModel', task_name, merged_res.sample_size)
            task_0 = merged_res.eval_map('RandomModel_0', task_name, merged_res.sample_size)
            task_1 = merged_res.eval_map('RandomModel_1', task_name, merged_res.sample_size)
            for res, res0, res1 in zip(task.res_list, task_0.res_list, task_1.res_list):
                if 'human_eval' not in res0 or 'human_eval' not in res1:
                    continue
                res['human_eval_0'], res['human_eval_1'] = res0['human_eval'], res1['human_eval']
                if 'human_eval_score' in res0:
                    res['human_eval_score_0'], res['human_eval_score_1'] = res0['human_eval_score'], res1['human_eval_score']
                if res0['human_eval'] == res1['human_eval']:
                    res['human_eval'] = res0['human_eval']
                    if 'human_eval_score' in res0:
                        res['human_eval_score'] = res0['human_eval_score']
                else:
                    res['human_eval'] = FAILED_TOKEN
                    if 'human_eval_score' in res0:
                        res['human_eval_score'] = FAILED_TOKEN
            task.save()

    def aggregate_human_evaluation(self):
        merged_res = EvalPipeline(model_name='RandomModel', cat=self.cat, sample_size=2)
        for task_name in merged_res.eval_list:
            task = merged_res.eval_map('RandomModel', task_name, merged_res.sample_size)
            res_list = [data for data in task.res_list if 'human_eval' in data and data['human_eval'] == FAILED_TOKEN]
            if len(res_list) != 0:
                interface = PreferenceInterface(data_list=res_list, mm_type=task_name[0])
                interface.start()
                idx = 0
                for data in res_list:
                    if interface.eval_list[idx] == 0:
                        data['human_eval'] = data['human_eval_0']
                        if 'human_eval_score' in data:
                            data['human_eval_score'] = data['human_eval_score_0']
                    else:
                        data['human_eval'] = data['human_eval_1']
                        if 'human_eval_score' in data:
                            data['human_eval_score'] = data['human_eval_score_1']
            task.save()

    def compute_inter_annotator_correlation(self):
        def normalize_text(text_list, language='english'):
            return_list = []
            for text in text_list:
                if language == 'english':
                    text = text.lower().strip()
                    text = unicodedata.normalize('NFD', text)
                    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
                    text = text.translate(str.maketrans('', '', string.punctuation))
                return_list.append(text or FAILED_TOKEN)
            return return_list

        merged_res = EvalPipeline(model_name='RandomModel', cat=self.cat, sample_size=2)
        corr_list = []
        for task_name in merged_res.eval_list:
            task = merged_res.eval_map('RandomModel', task_name, merged_res.sample_size)
            if 'human_eval' not in task.res_list[0]:
                corr_list.append(1.0)
                continue
            if 'human_eval_score' in task.res_list[0]:
                annotator0_list = [data['human_eval_score_0'] for data in task.res_list if 'human_eval_score_0' in data]
                annotator1_list = [data['human_eval_score_1'] for data in task.res_list if 'human_eval_score_1' in data]
                corr_list.append(calculate_agreement(annotator0_list, annotator1_list))
            else:
                annotator0_list = [[data['human_eval_0']] if not isinstance(data['human_eval_0'], list) else
                                   data['human_eval_0'] for data in task.res_list]
                annotator1_list = [[data['human_eval_1']] if not isinstance(data['human_eval_1'], list) else
                                   data['human_eval_1'] for data in task.res_list]
                if isinstance(annotator0_list[0][0], str):
                    import evaluate
                    if (any(re.findall(r'[\u4e00-\u9fff]', t) for tt in annotator0_list for t in tt) or
                            any(re.findall(r'[\u4e00-\u9fff]', t) for tt in annotator1_list for t in tt)):
                        language = 'chinese'
                    else:
                        language = 'english'
                    wer = evaluate.load('wer') if language == 'english' else evaluate.load('cer')
                    corr_list.append(np.mean([1.0 - min(wer.compute(predictions=normalize_text(a), references=normalize_text(b)), 1.0)
                                              for a, b in zip(annotator0_list, annotator1_list)]))
                else:
                    annotator0_cat = list(itertools.chain(*annotator0_list))
                    annotator1_cat = list(itertools.chain(*annotator1_list))
                    corr_list.append(calculate_agreement(annotator0_cat, annotator1_cat))
        merged_res.eval_df['inter_annotator_correlation'] = corr_list
        merged_res.eval_df.to_csv(f'./output/RandomModel/{self.cat}_eval.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Pipeline:')
    parser.add_argument('--model_name', type=str, default='Dalle3',
                        help='Name of the model. Make sure it is the same as your implemented class name.')
    parser.add_argument('--category', type=str, default='i', help='Subcategory of the benchmark: i, a, it, at.')
    parser.add_argument('--job', type=str, default='evaluate', help='Job type: generate, evaluate, human')
    args = parser.parse_args()

    pipeline = EvalPipeline(args.model_name, args.category, 4)
    if args.job == 'evaluate':
        pipeline.evaluate()
    elif args.job == 'human':
        pipeline.human_evaluate()
    else:
        pipeline.generate()

    # parser = argparse.ArgumentParser(description='Evaluation Benchmark:')
    # parser.add_argument('--category', type=str, default='asp', help='Subcategory of the benchmark: i, a, it, at.')
    # args = parser.parse_args()
    #
    # benchmark = EvalBenchmark(cat=args.category, sample_size=4)
    # benchmark.rank_models()
