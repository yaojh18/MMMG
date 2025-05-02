from eval_image import *
from eval_audio import *
from eval_interleaved import *
from utils import *

import argparse


class EvalPipeline:
    def __init__(self, model_name, cat='i', sample_size=1):
        assert cat in ['i', 'it', 'aso', 'asp', 'am']
        self.model_name = model_name
        self.cat = cat
        self.sample_size = sample_size
        if cat == 'i':
            self.eval_list = ['i_object_include', 'i_object_exclude', 'i_object_count', 'i_object_cot',
                              'i_object_attribute', 'i_relation_two', 'i_relation_all', 'i_spacial_relative',
                              'i_spacial_absolute', 'i_format_background', 'i_format_border', 'i_ocr',
                              'i_ocr_two', 'i_ocr_multi_lingual']
        if cat == 'it':
            self.eval_list = ['i_consistency_semantic', 'i_consistency_3d_object', 'i_consistency_3d_scene',
                              'i_consistency_compose', 'i_consistency_decompose', 'i_edit_add', 'i_edit_color',
                              'i_edit_text', 'i_edit_object_add', 'i_edit_object_remove', 'i_edit_object_modify',
                              'it_coherence_count', 'it_coherence_color', 'it_coherence_size', 'it_coherence_ocr',
                              'it_coherence_spacial_relative', 'it_coherence_spacial_absolute',
                              'it_coherence_math', 'it_coherence_code', 'i_structure']
        elif cat == 'aso':
            self.eval_list = ['a_sound_begin_end', 'a_sound_include', 'a_sound_cot', 'a_sound_silence']
        elif cat == 'asp':
            self.eval_list = ['a_speech_attribute', 'a_speech_chinese', 'a_speech_imitate', 'a_speech_modify',
                              'a_consistency_conversation', 'a_consistency_variant', 'a_structure']
        elif cat == 'am':
            self.eval_list = ['a_music_instrument', 'a_music_exclude', 'a_music_tempo', 'a_music_intensity']

        if os.path.exists(f'./output/{model_name}/{cat}_eval.csv'):
            self.eval_df = pd.read_csv(f'./output/{model_name}/{cat}_eval.csv')
        else:
            self.eval_df = pd.DataFrame({
                'task': self.eval_list,
                'accuracy': [None] * len(self.eval_list),
                'human_accuracy': [None] * len(self.eval_list),
                'correlation': [None] * len(self.eval_list)
            })
            os.makedirs(f'./output/{model_name}/', exist_ok=True)
            self.eval_df.to_csv(f'./output/{model_name}/{cat}_eval.csv', index=False)

    def eval_map(self, task_name):
        task_name = ''.join([t.capitalize() if i > 0 else t.upper() for i, t in enumerate(task_name.split('_'))])
        task_name = task_name.replace('Ocr', 'OCR').replace('Cot', 'CoT').replace('3d', '3D')
        return eval(f"{task_name}(model_name='{self.model_name}', sample_size={self.sample_size})")

    def evaluate(self):
        for index, row in self.eval_df.iterrows():
            if pd.isna(row['accuracy']):
                task_name = row['task']
                task = self.eval_map(task_name)
                # task.evaluate()
                # self.eval_df.loc[index, 'accuracy'] = task.compute_accuracy()
                # self.eval_df.to_csv(f'./output/{self.model_name}/{self.cat}_eval.csv', index=False)

    def human_evaluate(self):
        for index, row in self.eval_df.iterrows():
            task_name = row['task']
            task = self.eval_map(task_name)
            # task.evaluate()
            if pd.isna(row['human_accuracy']):
                task.human_evaluate()
            self.eval_df.loc[index, 'accuracy'] = task.compute_accuracy()
            self.eval_df.loc[index, ['human_accuracy', 'correlation']] = task.compute_correlation()
            self.eval_df.to_csv(f'./output/{self.model_name}/{self.cat}_eval.csv', index=False)


class EvalBenchmark:
    def __init__(self, model_list=[], cat='i', sample_size=4):
        assert cat in ['i', 'it', 'aso', 'asp', 'am']
        self.cat = cat
        self.sample_size = sample_size
        self.pipelines = {}

        if cat == 'i':
            base_model_list = ['Imagen3', 'Recraft3', 'LumaPhoton', 'Flux1_1Pro', 'Ideogram2', 'Dalle3',
                               'StableDiffusion3_5', 'GPT4o']
        elif cat == 'it':
            base_model_list = ['Gemini2', 'GeminiAgent', 'HybridAgent']
        elif cat == 'aso':
            base_model_list = ['StableAudio', 'AudioLDM2', 'AudioGen', 'MakeAnAudio2', 'Tango2']
        elif cat == 'asp':
            base_model_list = ['VoxInstructAgent', 'VoiceLDMAgent', 'SpiritLM', 'BaichuanAudio']
        else:
            base_model_list = ['StableAudio', 'AudioLDM2', 'MusicGen', 'TangoMusic', 'YuE']
        for model_name in base_model_list + model_list:
            pipeline = EvalPipeline(model_name, self.cat, self.sample_size)
            pipeline.evaluate()
            self.pipelines[model_name] = pipeline

    def rank_models(self, method='absolute'):
        reshaped_dfs = []
        for model_name, pipeline in self.pipelines.items():
            temp_df = pipeline.eval_df[['task', 'accuracy']].copy()
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

    def aggregate_human_evaluation(self):
        pass


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Evaluation Pipeline:')
    # parser.add_argument('--model_name', type=str, default='RandomModel_0', help='Name of the model.')
    # parser.add_argument('--category', type=str, default='it', help='Subcategory of the benchmark: i, a, it, at.')
    # parser.add_argument('--sample_size', type=int, default=2, help='Sample number of each instruction.')
    # args = parser.parse_args()
    #
    # pipeline = EvalPipeline(args.model_name, args.category, args.sample_size)
    # pipeline.evaluate()

    parser = argparse.ArgumentParser(description='Evaluation Benchmark:')
    parser.add_argument('--category', type=str, default='it', help='Subcategory of the benchmark: i, a, it, at.')
    parser.add_argument('--sample_size', type=int, default=4, help='Sample number of each instruction.')
    args = parser.parse_args()

    benchmark = EvalBenchmark(cat=args.category, sample_size=args.sample_size)
    benchmark.human_evaluate()
