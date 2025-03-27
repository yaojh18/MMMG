import itertools

from eval import EvalUnit
from prompt import *
from interface import *


class IConsistencySemantic(EvalUnit):
    inst_name = 'i_consistency_semantic'

    def evaluate(self):
        query_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != len(inst['object']):
                data['model_eval'] = [FAILED_TOKEN] * len(inst['object'])
                continue
            else:
                data['model_eval'] = []
            for image, target in zip(data['image_list'], inst['object']):
                query_list.append(form_openai_mm_query(IMAGE_TOKEN(0) + I_OBJECT_EXIST_PROMPT(target), images=[image]))
                data['model_eval'].append(idx)
                idx += 1

        responses = batch(query_openai, query_list, model='chatgpt-4o-latest', temperature=0.0)
        for data in self.res_list:
            data['model_eval'] = [float('yes' in responses[i].lower()) if i != FAILED_TOKEN else 0.0 for i in data['model_eval']]
        self.save()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != len(inst['object']):
                data['human_eval'] = [FAILED_TOKEN] * len(inst['object'])
                continue
            else:
                data['human_eval'] = []
            for image, target in zip(data['image_list'], inst['object']):
                human_inst_list.append(f"Is/Are there {target} in the given image?")
                human_res_list.append({"image_list": [image]})
                data['human_eval'].append(idx)
                idx += 1
        interface = MultiLabelInterface(
            label_list=("Yes", "No"),
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = [1.0 - interface.eval_list[i] if i != FAILED_TOKEN else 0.0 for i in data['human_eval']]
        self.save()

    def calculate_metrics(self):
        gpt_eval_list = [np.mean(res['model_eval']) for res in self.res_list]
        human_eval_list = [np.mean(res['human_eval']) for res in self.res_list]
        print(f"Model evaluation accuracy for {self.inst_name}: ", np.mean(gpt_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        gpt_eval_list = np.concatenate([res['model_eval'] for res in self.res_list])
        human_eval_list = np.concatenate([res['human_eval'] for res in self.res_list])
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(gpt_eval_list, human_eval_list))
        print(f"Agreement for {self.inst_name}: ", calculate_agreement(gpt_eval_list, human_eval_list))


class IConsistency3D(EvalUnit):
    inst_name = 'i_consistency_3d'

    def evaluate(self):
        self.load_inst_mm()
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != len(inst['ref_image_list']):
                data['auto_eval'] = [0.0] * len(inst['ref_image_list'])
            else:
                data['auto_eval'] = [calculate_dreamsim(img1, img2) for img1, img2 in zip(data['image_list'], inst['ref_image_list'])]
        self.save()

    def calculate_metrics(self):
        print(f'Auto evaluation accuracy for {self.inst_name}: ', np.mean([np.mean(data['auto_eval']) for data in self.res_list]))


class AConsistencyConversation(EvalUnit):
    inst_name = 'a_consistency_conversation'
    def evaluate(self):
        # Transcribe the script
        transcripts, _ = transcribe_speech(list(itertools.chain(*[data['audio_list'] for data in self.res_list])))
        idx = 0
        for data in self.res_list:
            data['transcript'] = transcripts[idx: idx + len(data['audio_list'])]
            idx += len(data['audio_list'])
        self.save()

        # Verify the text constraint
        text_list = []
        instruction_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['transcript']) != len(inst['order']):
                data['auto_eval'] = [FAILED_TOKEN] * len(inst['constraints'])
                continue
            else:
                data['auto_eval'] = []
            for key, val in inst['constraints'].items():
                text_list.append(data['transcript'][int(key)])
                instruction_list.append(val)
                data['auto_eval'].append(idx)
                idx += 1
        auto_eval_list = text_instruction_following_verify(text_list, instruction_list)
        for data in self.res_list:
            data['auto_eval'] = [auto_eval_list[i] if i != FAILED_TOKEN else 0.0 for i in data['auto_eval']]
        self.save()

        # Verify speaker similarity
        audio_list = []
        ref_audio_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            last_seen = {}
            pair_list = []
            for i, num in enumerate(inst['order']):
                if num in last_seen:
                    pair_list.append((last_seen[num], i))
                last_seen[num] = i
            if len(data['transcript']) != len(inst['order']):
                data['model_eval'] = [FAILED_TOKEN] * len(pair_list)
            else:
                data['model_eval'] = list(range(idx, idx + len(pair_list)))
                audio_list += [data['audio_list'][p[1]] for p in pair_list]
                ref_audio_list += [data['audio_list'][p[0]] for p in pair_list]
                idx += len(pair_list)
        scores = calculate_speech_similarity(audio_list, ref_audio_list)
        for data in self.res_list:
            data['model_eval'] = [scores[i] if i != FAILED_TOKEN else 0.0 for i in data['model_eval']]
        self.save()

    def human_evaluate(self):
        # Verify speaker similarity
        audio_list = []
        ref_audio_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            last_seen = {}
            pair_list = []
            for i, num in enumerate(inst['order']):
                if num in last_seen:
                    pair_list.append((last_seen[num], i))
                last_seen[num] = i
            if len(data['transcript']) != len(inst['order']):
                data['human_eval'] = [FAILED_TOKEN] * len(pair_list)
            else:
                data['human_eval'] = list(range(idx, idx + len(pair_list)))
                audio_list += [{"audio_list": [data['audio_list'][p[1]]]} for p in pair_list]
                ref_audio_list += [data['audio_list'][p[0]] for p in pair_list]
                idx += len(pair_list)
        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=['Are the speeches coming from the same speaker?'] * len(audio_list),
            data_list=audio_list,
            ref_list=ref_audio_list,
            mm_type='a'
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = [1.0 - interface.eval_list[i] if i != FAILED_TOKEN else 0.0 for i in data['human_eval']]
        self.save()

    def calculate_metrics(self, threshold=0.86):
        print(f'Auto evaluation accuracy for {self.inst_name}: ', np.mean([np.min(data['auto_eval']) for data in self.res_list]))
        model_eval_list = [data['model_eval'] for data in self.res_list]
        human_eval_list = [data['human_eval'] for data in self.res_list]

        model_eval_cat = list(itertools.chain(*model_eval_list))
        human_eval_cat = list(itertools.chain(*human_eval_list))
        model_eval_cat = [e > threshold for e in model_eval_cat]
        model_eval_list = [[e > threshold for e in me] for me in model_eval_list]

        print(f"Model evaluation accuracy for {self.inst_name}: ", np.mean([np.mean(e) for e in model_eval_list]))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean([np.mean(e) for e in human_eval_list]))
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(model_eval_cat, human_eval_cat))
        print(f"Agreement for {self.inst_name}: ", calculate_agreement(model_eval_cat, human_eval_cat))


class IStructure(EvalUnit):
    inst_name = 'i_structure'

    def evaluate(self):
        text_pattern = r'<(?:image|audio)_start><(?:image|audio)_\d+><(?:image|audio)_end>'
        mm_pattern = r'<((?:image|audio)_\d+)>'
        for data, inst in zip(self.res_list, self.inst_list):
            texts = re.split(text_pattern, data['response'])
            modalities = re.findall(mm_pattern, data['response'])
            mm_list = '' if texts[0] == '' else 't'
            for t, mm in zip(texts[1:], modalities):
                if mm.startswith('image'):
                    mm_list += 'i'
                else:
                    mm_list += 'a'
                if t.strip() != '':
                    mm_list += 't'
            data['auto_eval'] = float(mm_list in inst['order'])
        self.save()

    def calculate_metrics(self):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        print(f"Structure accuracy for {self.inst_name}: ", np.mean(auto_eval_list))


class AStructure(IStructure):
    inst_name = 'a_structure'


if __name__ == '__main__':
    a = IConsistency3D(model_name='ImageAgent', sample_size=2)
    a.evaluate()
    a.calculate_metrics()
