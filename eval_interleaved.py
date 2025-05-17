import itertools

from eval import EvalUnit
from prompt import *
from interface import *
from eval_image import IOCR


class IConsistencySemantic(EvalUnit):
    inst_name = 'i_consistency_semantic'

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != len(inst['object']):
                data['model_eval'] = [FAILED_TOKEN] * len(inst['object'])
                continue
            else:
                data['model_eval'] = []
            for image, target in zip(data['image_list'], inst['object']):
                queries.append(form_mm_query(I_OBJECT_EXIST_COT_PROMPT(target), images=[image]))
                data['model_eval'].append(idx)
                idx += 1

        responses = query_vlm(queries)
        for data in self.res_list:
            data['model_eval'] = [float('yes' in responses[i].strip().lower()[-20:]) if i != FAILED_TOKEN else 0.0
                                  for i in data['model_eval']]
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

    def compute_accuracy(self, return_list=False):
        model_eval_list = [np.prod(res['model_eval']) for res in self.res_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self):
        human_eval_list = [res['human_eval'] for res in self.res_list]
        model_eval_list = [res['model_eval'] for res in self.res_list]
        human_eval_cat = list(itertools.chain(*human_eval_list))
        model_eval_cat = list(itertools.chain(*model_eval_list))
        return calculate_agreement(model_eval_cat, human_eval_cat), calculate_pearson(model_eval_cat, human_eval_cat)


class IConsistencyCompose(IConsistencySemantic):
    inst_name = 'i_consistency_compose'


class IConsistencyDecompose(IConsistencySemantic):
    inst_name = 'i_consistency_decompose'


class IConsistency3DObject(EvalUnit):
    inst_name = 'i_consistency_3d_object'

    def evaluate(self):
        self.load_inst_mm()
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != len(inst['ref_image_list']):
                data['auto_eval'] = [0.0] * len(inst['ref_image_list'])
            else:
                data['auto_eval'] = [calculate_ssim(img1, img2) for img1, img2 in zip(data['image_list'], inst['ref_image_list'])]
        self.save()

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [np.mean(data['auto_eval']) for data in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


class IConsistency3DScene(IConsistency3DObject):
    inst_name = 'i_consistency_3d_scene'


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
                data['auto_eval'] = [FAILED_TOKEN] * len(inst['constraint'])
                continue
            else:
                data['auto_eval'] = []
            for key, val in inst['constraint'].items():
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
            pair_list = []
            for i, num in enumerate(inst['order']):
                for j, num2 in enumerate(inst['order'][i + 1:]):
                    if num == num2:
                        pair_list.append((i, j + i + 1))
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
            pair_list = []
            for i, num in enumerate(inst['order']):
                for j, num2 in enumerate(inst['order'][i + 1:]):
                    if num == num2:
                        pair_list.append((i, j + i + 1))
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

    def compute_accuracy(self, threshold=0.93, return_list=False):
        auto_eval_list = [np.mean(data['auto_eval']) for data in self.res_list]
        model_eval_list = [np.prod([me > threshold for me in data['model_eval']]) for data in self.res_list]
        combined_list = [a * m for a, m in zip(auto_eval_list, model_eval_list)]
        if return_list:
            return combined_list
        return np.mean(combined_list)

    def compute_correlation(self, threshold=0.93):
        model_eval_list = [res['model_eval'] for res in self.res_list]
        human_eval_list = [res['human_eval'] for res in self.res_list]

        # # Optimal threshold
        # model_eval_cat = list(itertools.chain(*model_eval_list))
        # human_eval_cat = list(itertools.chain(*human_eval_list))
        # threshold = find_optimal_threshold(model_eval_cat, human_eval_cat)

        model_eval_list = [[e > threshold for e in model_eval] for model_eval in model_eval_list]
        human_eval_cat = list(itertools.chain(*human_eval_list))
        model_eval_cat = list(itertools.chain(*model_eval_list))
        return calculate_agreement(model_eval_cat, human_eval_cat), calculate_pearson(model_eval_cat, human_eval_cat)


class IStructure(EvalUnit):
    inst_name = 'i_structure'

    def evaluate(self):
        text_pattern = r'<(?:image|audio)_start><(?:image|audio)_\d+><(?:image|audio)_end>'
        mm_pattern = r'<((?:image|audio)_\d+)>'
        for data, inst in zip(self.res_list, self.inst_list):
            texts = re.split(text_pattern, data['response'])
            modalities = re.findall(mm_pattern, data['response'])
            if len(texts) != len(modalities) + 1:
                data['auto_eval'] = 0.0
                continue
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

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


class AStructure(IStructure):
    inst_name = 'a_structure'


class ITCoherence(EvalUnit):
    idx = 0
    label_list: tuple
    default_eval = 0.0
    allow_multi_images = False
    vlm = 'openai'

    def evaluate(self):
        text_pattern = r'<image_start><image_\d+><image_end>'
        queries = []
        self.idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            texts = re.split(text_pattern, data['response'])
            if self.allow_multi_images:
                if len(texts) < 2 or len(data['image_list']) < 1:
                    data['model_eval'] = FAILED_TOKEN
                    continue
            else:
                if len(texts) != 2 or len(data['image_list']) != 1:
                    data['model_eval'] = FAILED_TOKEN
                    continue
            # texts = [t.strip() for t in texts if t.strip() != '']
            # if len(texts) != 1:
            #     data['model_eval'] = FAILED_TOKEN
            #     continue
            # self.model_process_data(data, inst, texts[0], queries)
            self.model_process_data(data, inst, ''.join(texts), queries)
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            if data['model_eval'] != FAILED_TOKEN:
                self.model_process_response(data, responses)
            else:
                data['model_eval'] = self.default_eval
        self.save()

    def model_process_data(self, data, inst, res, queries):
        pass

    @staticmethod
    def model_process_response(data, responses):
        pass

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        self.idx = 0
        for data in self.res_list:
            self.human_process_data(data, human_inst_list, human_res_list)
        interface = MultiLabelInterface(
            label_list=self.label_list,
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            if data['human_eval'] != FAILED_TOKEN:
                self.human_process_response(data, interface.eval_list)
            else:
                data['human_eval'] = self.default_eval
        self.save()

    def human_process_data(self, data, human_inst_list, human_res_list):
        pass

    @staticmethod
    def human_process_response(data, responses):
        pass

    def compute_accuracy(self, return_list=False):
        model_eval_list = [np.prod(res['model_eval']) for res in self.res_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self):
        model_eval_list = [res['model_eval'] if isinstance(res['model_eval'], list) else [res['model_eval']] for res in self.res_list]
        human_eval_list = [res['human_eval'] if isinstance(res['human_eval'], list) else [res['human_eval']] for res in self.res_list]
        human_eval_cat = list(itertools.chain(*human_eval_list))
        model_eval_cat = list(itertools.chain(*model_eval_list))
        return calculate_agreement(model_eval_cat, human_eval_cat), calculate_pearson(model_eval_cat, human_eval_cat)


class ITCoherenceCount(ITCoherence):
    label_list = ("A. Less than 3", "B. 3", "C. 4", "D. 5", "E. 6", "F. More than 6")
    inst_name = 'it_coherence_count'

    def model_process_data(self, data, inst, res, queries):
        res = re.search(r'<count>(\d)+</count>', res)
        if res is None or not (2 < int(res.group(1)) < 7):
            data['model_eval'] = FAILED_TOKEN
            return
        cnt = int(res.group(1))
        queries.append(form_mm_query(I_OBJECT_COUNT_PROMPT(inst['object']), images=data['image_list'], model=self.vlm))
        data['object'], data['count'], data['model_eval'] = inst['object'], cnt, self.idx
        self.idx += 1

    @staticmethod
    def model_process_response(data, responses):
        data['model_eval'] = float(ord(responses[data['model_eval']].strip().lower()[0]) - 95 == data['count'])

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'object' not in data:
            data['human_eval'] = FAILED_TOKEN
            return
        human_inst_list.append(f"How many {data['object']} are there in the given image?\n")
        human_res_list.append(data)
        data['human_eval'] = self.idx
        self.idx += 1

    @staticmethod
    def human_process_response(data, responses):
        data['human_eval'] = float(responses[data['human_eval']] + 2 == data['count'])


class ITCoherenceColor(ITCoherence):
    label_list = ('Yes', 'No')
    inst_name = 'it_coherence_color'
    default_eval = [0.0, 0.0, 0.0]
    exclude_list = ['microwave', 'ball', 'TV', 'globe', 'toothbrush', 'knife', 'tennis racket', 'camera']

    def model_process_data(self, data, inst, res, queries):
        obj2col = extract_json(res)
        if not obj2col or (set(obj2col.keys()) != set(inst['object'])) or (set(obj2col.values()) != set(inst['color'])):
            data['model_eval'] = FAILED_TOKEN
            return
        queries += [form_mm_query(
            I_OBJECT_EXIST_COT_PROMPT(
                f'exactly one {obj} and the color of {obj} being majorly {col} and no other major color'
                if obj not in self.exclude_list else f'exactly one {obj} and the color of {obj} being majorly {col}'
            ),
            images=data['image_list'], model=self.vlm
        ) for obj, col in obj2col.items()]
        data['obj2col'], data['model_eval'] = obj2col, [self.idx, self.idx + 1, self.idx + 2]
        self.idx += 3

    @staticmethod
    def model_process_response(data, responses):
        data['model_eval'] = [float('yes' in responses[i].strip().lower()[-20:]) for i in data['model_eval']]

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'obj2col' not in data:
            data['human_eval'] = FAILED_TOKEN
            return
        human_inst_list += [f"Is/Are there exactly one {obj} and the color of {obj} being mostly {col} in the given image?\n"
                            for obj, col in data['obj2col'].items()]
        human_res_list += [data] * 3
        data['human_eval'] = [self.idx, self.idx + 1, self.idx + 2]
        self.idx += 3

    @staticmethod
    def human_process_response(data, responses):
        data['human_eval'] = [float(responses[i] == 0) for i in data['human_eval']]


class ITCoherenceSize(ITCoherenceColor):
    inst_name = 'it_coherence_size'

    def model_process_data(self, data, inst, res, queries):
        obj_list = extract_list(res)
        if not obj_list or set(obj_list) != set(inst['object']):
            data['model_eval'] = FAILED_TOKEN
            return
        rel_map = {'size': 'larger', 'area': 'larger', 'volume': 'bigger', 'length': 'longer', 'height': 'higher'}
        queries += [form_mm_query(I_OBJECT_EXIST_COT_PROMPT(
            f"exactly one {obj_list[i]}, exactly one {obj_list[j]} and the {obj_list[j]} being obviously "
            f"{rel_map[inst['relation']]} than the {obj_list[i]}") + (inst['text'] if 'text' in inst else ''),
            images=data['image_list'], model=self.vlm) for i in range(3) for j in range(i + 1, 3)]
        data['object'], data['relation'] = obj_list, rel_map[inst['relation']]
        data['model_eval'] = [self.idx, self.idx + 1, self.idx + 2]
        self.idx += 3

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'object' not in data:
            data['human_eval'] = FAILED_TOKEN
            return
        human_inst_list += [(f"Is/Are there exactly one {data['object'][i]}, exactly one {data['object'][j]} and the "
                             f"{data['object'][j]} being obviously {data['relation']} than the "
                             f"{data['object'][i]} in the given image?\n") for i in range(3) for j in range(i + 1, 3)]
        human_res_list += [data] * 3
        data['human_eval'] = [self.idx, self.idx + 1, self.idx + 2]
        self.idx += 3


class ITCoherenceSpacialRelative(ITCoherenceColor):
    inst_name = 'it_coherence_spacial_relative'
    default_eval = [0.0, 0.0]
    vlm = 'gemini'

    def evaluate(self):
        super().evaluate()
        self.idx = 0
        queries = []
        for data in self.res_list:
            if data['model_eval'][0] == 1.0:
                queries.append(form_mm_query(I_SPACIAL_RELATIVE_LR(
                    data['object'][0][0], data['object'][0][1]), images=data['image_list'], model=self.vlm))
                data['model_eval'][0] = self.idx
                self.idx += 1
            if data['model_eval'][1] == 1.0:
                queries.append(form_mm_query(I_SPACIAL_RELATIVE_UD(
                    data['object'][1][0], data['object'][1][1]), images=data['image_list'], model=self.vlm))
                data['model_eval'][1] = self.idx
                self.idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            if isinstance(data['model_eval'][0], int):
                opt = re.search('nswer: ([ABC])', responses[data['model_eval'][0]])
                if opt is not None:
                    data['model_eval'][0] = float(ord(opt.group(1)) - 65 == data['object'][0][2])
                else:
                    data['model_eval'][0] = 0.0
            if isinstance(data['model_eval'][1], int):
                opt = re.search('nswer: ([ABC])', responses[data['model_eval'][1]])
                if opt is not None:
                    data['model_eval'][1] = float(ord(opt.group(1)) - 65 == data['object'][1][2])
                else:
                    data['model_eval'][1] = 0.0
        self.save()

    def model_process_data(self, data, inst, res, queries):
        opt_list = extract_list(res)
        opt_list = [obt.strip().lower() for obt in opt_list]
        if not opt_list or len(opt_list) != 2 or not (set(opt_list) < {'a', 'b', 'c'}):
            data['model_eval'] = FAILED_TOKEN
            return
        queries += [form_mm_query(I_OBJECT_EXIST_COT_PROMPT(f"exactly one {obj1} and exactly one {obj2}"),
                                  images=data['image_list'], model=self.vlm) for obj1, obj2 in inst['object']]
        data['model_eval'], data['object'] = ([self.idx, self.idx + 1],
                                              [a + [ord(b) - 97] for a, b in zip(inst['object'], opt_list)])
        self.idx += 2

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'object' not in data:
            data['human_eval'] = FAILED_TOKEN
            return
        lr_map = ['to the left of', 'to the right of', 'neither to the obviously left nor right of']
        human_inst_list.append(f"Is/Are there exactly one {data['object'][0][0]}, exactly one {data['object'][0][1]} "
                               f"and the {data['object'][0][0]} is {lr_map[data['object'][0][2]]} "
                               f"the {data['object'][0][1]} in the given image?\n")
        ud_map = ['positioned higher than', 'positioned lower than', 'neither positioned obviously higher nor lower than']
        human_inst_list.append(f"Is/Are there exactly one {data['object'][1][0]}, exactly one {data['object'][1][1]} "
                               f"and the {data['object'][1][0]} is {ud_map[data['object'][1][2]]} "
                               f"the {data['object'][1][1]} in the given image?\n")
        human_res_list += [data] * 2
        data['human_eval'] = [self.idx, self.idx + 1]
        self.idx += 2


class ITCoherenceSpacialAbsolute(ITCoherenceColor):
    inst_name = 'it_coherence_spacial_absolute'
    default_eval = [0.0, 0.0]
    vlm = 'gemini'

    def evaluate(self):
        super().evaluate()
        self.idx = 0
        queries = []
        for data in self.res_list:
            for i in range(2):
                if data['model_eval'][i] == 1.0:
                    queries.append(form_mm_query(I_SPACIAL_ABSOLUTE_PROMPT(data['object'][i][0]),
                                                 images=data['image_list'], model=self.vlm))
                    data['model_eval'][i] = self.idx
                    self.idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            for i in range(2):
                if isinstance(data['model_eval'][i], int):
                    opt = re.search('nswer: ([ABCDE])', responses[data['model_eval'][i]])
                    if opt is not None:
                        data['model_eval'][i] = float(ord(opt.group(1)) - 65 == data['object'][i][1])
                    else:
                        data['model_eval'][i] = 0.0
        self.save()

    def model_process_data(self, data, inst, res, queries):
        opt_list = extract_list(res)
        opt_list = [obt.strip().lower() for obt in opt_list]
        if not opt_list or len(opt_list) != 2 or not (set(opt_list) < {'a', 'b', 'c', 'd'}):
            data['model_eval'] = FAILED_TOKEN
            return
        queries += [form_mm_query(I_OBJECT_EXIST_COT_PROMPT(f"exactly one {obj}"),
                                  images=data['image_list'], model=self.vlm) for obj in inst['object']]
        data['model_eval'], data['object'] = ([self.idx, self.idx + 1],
                                              [[a, ord(b) - 97] for a, b in zip(inst['object'], opt_list)])
        self.idx += 2

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'object' not in data:
            data['human_eval'] = FAILED_TOKEN
            return
        rel_map = ['bottom left', 'bottom right', 'up left', 'up right']
        human_inst_list += [f"Is/Are there exactly one {obj} and at the {rel_map[rel]} of the given image?\n"
                            for obj, rel in data['object']]
        human_res_list += [data] * 2
        data['human_eval'] = [self.idx, self.idx + 1]
        self.idx += 2


class ITCoherenceOCR(ITCoherence, IOCR):
    inst_name = 'it_coherence_ocr'
    default_eval = ''

    def model_process_data(self, data, inst, res, queries):
        res = re.search(r'<text>(.*?)</text>', res)
        if res is None or not (5 <= len(res.group(1).split(' ')) <= 10):
            data['model_eval'] = FAILED_TOKEN
            return
        text = res.group(1)
        queries.append(form_mm_query(I_OCR_ENGLISH_PROMPT(inst['object']), images=data['image_list'], model=self.vlm))
        data['text'], data['model_eval'] = text, self.idx
        self.idx += 1

    @staticmethod
    def model_process_response(data, responses):
        data['model_eval'] = ' '.join(extract_list(responses[data['model_eval']])).lower().strip()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        self.idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if 'text' in data:
                human_inst_list.append(f"Please type the major texts (ignore small texts on the edge) on {inst['object']}"
                                       f" from top to down, from left to right in the given image. Leave empty if the there "
                                       f"is no valid Latin character in the given image. Ignore small texts in the corner.")
                human_res_list.append(data)
                data['human_eval'] = self.idx
                self.idx += 1
            else:
                data['human_eval'] = ''
        interface = FreeLabelInterface(
            eval_inst_list=human_inst_list,
            data_list=human_res_list
        )
        interface.start()
        for data in self.res_list:
            if data['human_eval'] != FAILED_TOKEN:
                data['human_eval'] = interface.eval_list[data['human_eval']].lower().strip()
        self.save()

    def compute_accuracy(self, return_list=False):
        label_list = [[self.normalize_text(res['text']) if 'text' in res else FAILED_TOKEN] for res in self.res_list]
        model_eval_list = [[self.normalize_text(res['model_eval']) if 'text' in res else FAILED_TOKEN] for res in self.res_list]
        mask_list = ['text' in res for res in self.res_list]
        wer_list = self._compute_accuracy(label_list, model_eval_list, return_list=True)
        wer_list = [wer if mask else 0.0 for mask, wer in zip(mask_list, wer_list)]
        if return_list:
            return wer_list
        return np.mean(wer_list)

    def compute_correlation(self):
        label_list = [[self.normalize_text(res['text'] if 'text' in res else '')] for res in self.res_list]
        model_eval_list = [[self.normalize_text(res['model_eval'])] for res in self.res_list]
        human_eval_list = [[self.normalize_text(res['human_eval'])] for res in self.res_list]
        return self._compute_correlation(label_list, model_eval_list, human_eval_list)


class ITCoherenceMath(ITCoherenceColor):
    inst_name = 'it_coherence_math'
    allow_multi_images = True
    vlm = 'gemini'
    default_eval = [0.0, 0.0]

    def evaluate(self):
        self.load_inst_mm()
        super().evaluate()

    def model_process_data(self, data, inst, res, queries):
        text = re.search(r"<<(.*?)>>", res)
        if text is None:
            data['model_eval'] = [0.0, self.idx]
            self.idx += 1
        else:
            data['model_eval'] = [self.idx, self.idx + 1]
            data['text'] = text.group(1)
            queries.append(form_mm_query(LLM_AS_A_JUDGE_PROMPT.format(inst['pattern'], text.group(1)), model=self.vlm))
            self.idx += 2
        queries.append(form_mm_query(VLM_AS_A_JUDGE_PROMPT.format(inst['pattern']), images=data['image_list'][-1:], model=self.vlm))
        data['pattern'] = inst['pattern']

    @staticmethod
    def model_process_response(data, responses):
        data['model_eval'] = [float('yes' in responses[i].strip().lower()[-20:]) if isinstance(i, int) else i
                              for i in data['model_eval']]

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'text' in data:
            human_inst_list.append("(Ignore the given image.)\n" + LLM_AS_A_JUDGE_PROMPT.format(data['pattern'], data['text']))
            human_res_list.append({'image_list': data['image_list'][-1:]})
            data['human_eval'] = [self.idx, 0.0]
            self.idx += 1
        else:
            data['human_eval'] = [0.0, 0.0]
        if 'pattern' in data:
            human_inst_list.append(VLM_AS_A_JUDGE_PROMPT.format(data['pattern']))
            human_res_list.append({'image_list': data['image_list'][-1:]})
            data['human_eval'][1] = self.idx
            self.idx += 1

    @staticmethod
    def human_process_response(data, responses):
        data['human_eval'] = [float(responses[i] == 0) if isinstance(i, int) else i for i in data['human_eval']]


class ITCoherenceCode(ITCoherenceColor):
    inst_name = 'it_coherence_code'
    allow_multi_images = True
    default_eval = 0.0

    def evaluate(self):
        super().evaluate()
        self.load_inst_mm()
        text_pattern = r'<image_start><image_\d+><image_end>'
        for data, inst in zip(self.res_list, self.inst_list):
            texts = re.split(text_pattern, data['response'])
            if len(texts) < 2 or len(data['image_list']) < 1:
                data['auto_eval'] = 0.0
            else:
                data['auto_eval'] = calculate_dreamsim(data['image_list'][-1], inst['ref_image_list'][0])
        self.save()

    def model_process_data(self, data, inst, res, queries):
        queries.append(form_mm_query(I_OBJECT_EXIST_COT_PROMPT(inst['object']), images=data['image_list'][-1:], model=self.vlm))
        data['object'], data['model_eval'] = inst['object'], self.idx
        self.idx += 1

    @staticmethod
    def model_process_response(data, responses):
        data['model_eval'] = float('yes' in responses[data['model_eval']].strip().lower()[-20:])

    def human_process_data(self, data, human_inst_list, human_res_list):
        if 'object' in data:
            human_inst_list.append(f"Is/Are there {data['object']} in the given image?\n")
            human_res_list.append(data)
            data['human_eval'] = self.idx
            self.idx += 1
        else:
            data['human_eval'] = FAILED_TOKEN

    @staticmethod
    def human_process_response(data, responses):
        data['human_eval'] = float(responses[data['human_eval']] == 0)

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [res['auto_eval'] * res['model_eval'] for res in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


if __name__ == '__main__':
    pass
