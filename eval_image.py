import unicodedata

from eval import *
from prompt import *


class IObject(EvalUnit):
    label_list: tuple
    vlm: str

    @staticmethod
    @abstractmethod
    def instruction_func(obj: str):
        pass

    @staticmethod
    @abstractmethod
    def human_instruction_func(obj: str):
        pass

    @staticmethod
    @abstractmethod
    def gpt_judge_process_func(res: str):
        pass

    @staticmethod
    @abstractmethod
    def human_judge_process_func(res: str):
        pass

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            obj_list = [inst['object']] if isinstance(inst['object'], str) else inst['object']
            if len(data['image_list']) != 1:
                data['model_eval'] = [FAILED_TOKEN] * len(obj_list)
                continue
            if self.inst_name == 'i_object_include' or self.inst_name == 'i_object_exclude':
                queries.append(form_mm_query(I_SCENE_PROMPT(obj_list[0]), images=data['image_list'], model=self.vlm))
                queries += [form_mm_query(self.instruction_func(obj), images=data['image_list'], model=self.vlm)
                            for obj in obj_list[1:]]
            else:
                queries += [form_mm_query(self.instruction_func(obj), images=data['image_list'], model=self.vlm)
                            for obj in obj_list]
            data['model_eval'] = list(range(idx, idx + len(obj_list)))
            idx += len(obj_list)
        responses = query_vlm(queries, model=self.vlm)
        parsed_responses = [self.gpt_judge_process_func(res) for res in responses]
        for data in self.res_list:
            data['model_eval'] = [parsed_responses[idx] if idx != FAILED_TOKEN else 0.0 for idx in data['model_eval']]
        self.save()

        if self.inst_name == 'i_object_count':
            for data, inst in zip(self.res_list, self.inst_list):
                data['model_eval'] = [float(data['model_eval'][0] == (inst['count'] - 2))]
            self.save()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            obj_list = [inst['object']] if isinstance(inst['object'], str) else inst['object']
            if len(data['image_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            human_inst_list.append(self.human_instruction_func(obj_list))
            human_res_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = MultiLabelInterface(
            label_list=self.label_list,
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = self.human_judge_process_func(interface.eval_list[data['human_eval']]) \
                if data['human_eval'] != FAILED_TOKEN else 0.0
        self.save()

        if self.inst_name == 'i_object_count':
            for data, inst in zip(self.res_list, self.inst_list):
                data['human_eval'] = [float(data['human_eval'] == (inst['count'] - 2))]
            self.save()

    def compute_accuracy(self, return_list=False):
        model_eval_list = [np.mean(res['model_eval']) for res in self.res_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self):
        human_eval_list = [np.mean(res['human_eval']) for res in self.res_list]
        model_eval_list = [np.mean(res['model_eval']) for res in self.res_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class IObjectInclude(IObject):
    inst_name = 'i_object_include'
    vlm = 'openai'
    label_list = ('Yes', 'No')

    @staticmethod
    def instruction_func(obj):
        return I_OBJECT_EXIST_COT_PROMPT(obj)

    @staticmethod
    def human_instruction_func(obj_list):
        if len(obj_list) == 1:
            return f"Is/Are there {obj_list[0]} in the given image?\n"
        else:
            return f"Is the given image about {obj_list[0]} and is/are there {', '.join(obj_list[1:])} in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if float('yes' in res.strip().lower()[-20:]) else 0.0

    @staticmethod
    def human_judge_process_func(res: str):
        return 1.0 if res == 0 else 0.0

    def compute_accuracy(self, return_list=False):
        if self.inst_name != 'i_object_include':
            return super().compute_accuracy(return_list)
        model_eval_list = [float(res['model_eval'][0] == 1.0) * np.mean(res['model_eval'][1:]) for res in self.res_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self):
        if self.inst_name != 'i_object_include':
            return super().compute_correlation()
        human_eval_list = [res['human_eval'] for res in self.res_list]
        model_eval_list = [float(res['model_eval'][0] == 1.0) * np.mean(res['model_eval'][1:]) for res in self.res_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class IObjectAttribute(IObjectInclude):
    inst_name = 'i_object_attribute'
    

class IObjectExclude(IObjectInclude):
    inst_name = 'i_object_exclude'

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if float('no' in res.strip().lower()[-20:]) else 0.0

    @staticmethod
    def human_instruction_func(obj_list):
        if len(obj_list) == 1:
            return f"Is/Are there {obj_list[0]} NOT in the given image?\n"
        else:
            return f"Is the given image about {obj_list[0]} and is/are there {', '.join(obj_list[1:])} NOT in the given image?\n"

    def compute_accuracy(self, return_list=False):
        if self.inst_name != 'i_object_exclude':
            return super().compute_accuracy(return_list)
        model_eval_list = [float(res['model_eval'][0] == 0.0) * np.mean(res['model_eval'][1:]) for res in self.res_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self):
        if self.inst_name != 'i_object_exclude':
            return super().compute_correlation()
        human_eval_list = [res['human_eval'] for res in self.res_list]
        model_eval_list = [float(res['model_eval'][0] == 0.0) * np.mean(res['model_eval'][1:]) for res in self.res_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class IObjectCoT(IObjectInclude):
    inst_name = 'i_object_cot'
    label_list = ("Yes", "No")

    @staticmethod
    def instruction_func(obj):
        return I_OBJECT_EXIST_COT_PROMPT(obj)

    @staticmethod
    def human_instruction_func(obj_list):
        return f"Is/Are there {obj_list[0]} in the given image?\n"


class IObjectCount(IObject):
    inst_name = 'i_object_count'
    vlm = 'openai'
    label_list = ("A. Less than 3", "B. 3", "C. 4", "D. 5", "E. 6", "F. More than 6")

    @staticmethod
    def instruction_func(obj):
        return I_OBJECT_COUNT_PROMPT(obj)

    @staticmethod
    def human_instruction_func(obj_list):
        return f"How many {obj_list[0]} are there in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return ord(res.strip().lower()[0]) - 97

    @staticmethod
    def human_judge_process_func(res: str):
        return res


class IRelationTwo(IObjectInclude):
    inst_name = 'i_relation_two'


class IRelationAll(IObjectInclude):
    inst_name = 'i_relation_all'


class ISpacial(EvalUnit):
    @staticmethod
    @abstractmethod
    def human_instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def human_judge_process_func(res: str):
        pass

    def human_evaluate(self):
        human_queries = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['human_eval'] = [FAILED_TOKEN] * len(inst['constraint'])
                continue
            human_res_list += [data] * len(inst['constraint'])
            for constraint in inst['constraint']:
                human_queries.append(self.human_instruction_func(constraint))
            data['human_eval'] = list(range(idx, idx + len(inst['constraint'])))
            idx += len(inst['constraint'])
        interface = MultiLabelInterface(
            label_list=("Yes", "No"),
            eval_inst_list=human_queries,
            data_list=human_res_list
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = [self.human_judge_process_func(interface.eval_list[idx]) if idx != FAILED_TOKEN
                                  else 0.0 for idx in data['human_eval']]
        self.save()

    def compute_accuracy(self, return_list=False):
        model_eval_list = [np.prod(res['model_eval']) for res in self.res_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self):
        human_eval_list = [np.prod(res['human_eval']) for res in self.res_list]
        model_eval_list = [np.prod(res['model_eval']) for res in self.res_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class ISpacialAbsolute(ISpacial):
    inst_name = 'i_spacial_absolute'
    vlm = 'gemini'

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['model_eval'] = [FAILED_TOKEN] * len(inst['constraint'])
                continue
            queries += [form_mm_query(I_OBJECT_EXIST_COT_PROMPT(f"exactly one {obj}"),
                                      model=self.vlm, images=data['image_list']) for obj, _ in inst['constraint']]
            data['model_eval'] = list(range(idx, idx + len(inst['constraint'])))
            idx += len(inst['constraint'])
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            data['model_eval'] = [float('yes' in responses[i].strip().lower()[-20:]) if i != FAILED_TOKEN
                                  else 0.0 for i in data['model_eval']]
        self.save()

        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            for i in range(len(inst['constraint'])):
                if data['model_eval'][i] == 1.0:
                    queries.append(form_mm_query(I_SPACIAL_ABSOLUTE_PROMPT(inst['constraint'][i][0]),
                                                 images=data['image_list'], model=self.vlm))
                    data['model_eval'][i] = idx
                    idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data, inst in zip(self.res_list, self.inst_list):
            for i in range(len(inst['constraint'])):
                if isinstance(data['model_eval'][i], int):
                    opt = re.search('nswer: ([ABCDE])', responses[data['model_eval'][i]])
                    rel = {'bottom left': 0, 'bottom right': 1, 'up left': 2, 'up right': 3}[inst['constraint'][i][1]]
                    if opt is not None:
                        data['model_eval'][i] = float(ord(opt.group(1)) - 65 == rel)
                    else:
                        data['model_eval'][i] = 0.0
        self.save()

    @staticmethod
    def human_instruction_func(const: dict):
        return f'Is there exactly one {const[0]} and at the {const[1]} of the given image?\n'

    @staticmethod
    def human_judge_process_func(res: str):
        return float(res == 0)


class ISpacialRelative(ISpacial):
    inst_name = 'i_spacial_relative'
    vlm = 'openai'

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['model_eval'] = [FAILED_TOKEN] * len(inst['constraint'])
                continue
            queries.append(form_mm_query(
                I_OBJECT_EXIST_COT_PROMPT(f"exactly one {inst['constraint'][0][0]} and exactly one {inst['constraint'][0][1]}"),
                images=data['image_list'], model=self.vlm))
            data['model_eval'] = [idx]
            idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            data['model_eval'] = [float('yes' in responses[i].strip().lower()[-20:]) if i != FAILED_TOKEN
                                  else 0.0 for i in data['model_eval']]
        self.save()

        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if data['model_eval'][0] == 1.0:
                if inst['constraint'][0][2] in ('left', 'right'):
                    queries.append(form_mm_query(
                        I_SPACIAL_RELATIVE_LR(inst['constraint'][0][0], inst['constraint'][0][1]),
                        images=data['image_list'], model=self.vlm))
                else:
                    queries.append(form_mm_query(
                        I_SPACIAL_RELATIVE_UD(inst['constraint'][0][0], inst['constraint'][0][1]),
                        images=data['image_list'], model=self.vlm))
                data['model_eval'] = [idx]
                idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data, inst in zip(self.res_list, self.inst_list):
            if isinstance(data['model_eval'][0], int):
                opt = re.search('nswer: ([ABC])', responses[data['model_eval'][0]])
                rel = {'left': 0, 'right': 1, 'above': 0, 'below': 1}[inst['constraint'][0][2]]
                if opt is not None:
                    data['model_eval'][0] = float(ord(opt.group(1)) - 65 == rel)
                else:
                    data['model_eval'][0] = 0.0
        self.save()

    @staticmethod
    def human_instruction_func(const: dict):
        rel = {'left': 'to the left of', 'right': 'to the right of',
               'above': 'positioned higher than', 'below': 'positioned lower than'}[const[2]]
        return f'Is there exactly one {const[0]}, exactly one {const[1]} and {const[0]} is {rel} {const[1]} in the given image?\n'

    @staticmethod
    def human_judge_process_func(res: str):
        return float(res == 0)


class IOCR(EvalUnit):
    inst_name = 'i_ocr'
    vlm = 'openai'
    language = 'english'

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['model_eval'] = FAILED_TOKEN
                continue
            queries.append(form_mm_query(I_OCR_ENGLISH_PROMPT(inst['object']) if 'object' in inst else I_OCR_EDIT_PROMPT,
                                         images=data['image_list'], model=self.vlm))
            data['model_eval'] = idx
            idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            data['model_eval'] = ' '.join(extract_list(responses[data['model_eval']])).lower().strip() \
                if data['model_eval'] != FAILED_TOKEN else ''
        self.save()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            human_inst_list.append(f"Please type the major texts (ignore small texts on the edge) on {inst['object'] if 'object' in inst else 'the given image'} "
                                   f"from top to down, from left to right in the given image. Leave empty if the there "
                                   f"is no valid Latin character in the given image. Ignore small texts in the corner.")
            human_res_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = FreeLabelInterface(
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = interface.eval_list[data['human_eval']].lower().strip() \
                if data['human_eval'] != FAILED_TOKEN else ''
        self.save()

    def compute_accuracy(self, return_list=False):
        label_list = [[self.normalize_text(inst['text'])] for inst in self.inst_list]
        model_eval_list = [[self.normalize_text(res['model_eval'])] for res in self.res_list]
        return self._compute_accuracy(label_list, model_eval_list, return_list)

    def compute_correlation(self):
        label_list = [[self.normalize_text(inst['text'])] for inst in self.inst_list]
        model_eval_list = [[self.normalize_text(res['model_eval'])] for res in self.res_list]
        human_eval_list = [[self.normalize_text(res['human_eval'])] for res in self.res_list]
        return self._compute_correlation(label_list, model_eval_list, human_eval_list)

    @staticmethod
    def normalize_text(text):
        normalized_text = text.lower().strip()
        normalized_text = unicodedata.normalize('NFD', normalized_text)
        normalized_text = ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')
        normalized_text = normalized_text.translate(str.maketrans('', '', string.punctuation))
        return normalized_text or FAILED_TOKEN

    def _compute_accuracy(self, label_list, model_eval_list, return_list=False):
        import evaluate
        wer = evaluate.load('wer') if self.language == 'english' else evaluate.load('cer')
        wer_list = [1.0 - min(wer.compute(predictions=model_eval, references=label), 1.0)
                    for model_eval, label in zip(model_eval_list, label_list)]
        if return_list:
            return wer_list
        return np.mean(wer_list)

    def _compute_correlation(self, label_list, model_eval_list, human_eval_list):
        import evaluate
        wer = evaluate.load('wer') if self.language == 'english' else evaluate.load('cer')
        model_wer_list = [1.0 - min(wer.compute(predictions=model_eval, references=label), 1.0)
                          for model_eval, label in zip(model_eval_list, label_list)]
        human_wer_list = [1.0 - min(wer.compute(predictions=human_eval, references=label), 1.0)
                          for human_eval, label in zip(human_eval_list, label_list)]
        correlated_wer_list = [1.0 - min(wer.compute(predictions=model_eval, references=human_eval), 1.0)
                               for model_eval, human_eval in zip(model_eval_list, human_eval_list)]
        return np.mean(correlated_wer_list), calculate_pearson(model_wer_list, human_wer_list)


class IOCRTwo(IOCR):
    inst_name = 'i_ocr_two'

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['model_eval'] = FAILED_TOKEN
                continue
            queries.append(form_mm_query(I_OBJECT_EXIST_COT_PROMPT(inst['object']), images=data['image_list'], model=self.vlm))
            data['model_eval'] = idx
            idx += 1
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            if data['model_eval'] != FAILED_TOKEN and 'yes' in responses[data['model_eval']].strip().lower()[-20:]:
                data['model_eval'] = [0.0, 0.0]
            else:
                data['model_eval'] = [FAILED_TOKEN, FAILED_TOKEN]
        self.save()

        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if FAILED_TOKEN not in data['model_eval']:
                queries += [form_mm_query(I_OCR_ENGLISH_PROMPT(obj), images=data['image_list'], model=self.vlm)
                            for obj in inst['text'].keys()]
                data['model_eval'] = [idx, idx + 1]
                idx += 2
        responses = query_vlm(queries, model=self.vlm)
        for data in self.res_list:
            if FAILED_TOKEN not in data['model_eval']:
                for i in range(2):
                    data['model_eval'][i] = ' '.join(extract_list(responses[data['model_eval'][i]])).lower().strip()
            else:
                data['model_eval'] = ['', '']
        self.save()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            human_inst_list.append(f"Is/Are there {inst['object']} in the given image?")
            human_res_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            if data['human_eval'] != FAILED_TOKEN and interface.eval_list[data['human_eval']] == 0:
                data['human_eval'] = [0.0, 0.0]
            else:
                data['human_eval'] = [FAILED_TOKEN, FAILED_TOKEN]
        self.save()

        data_list = []
        eval_inst_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if FAILED_TOKEN not in data['human_eval']:
                data_list += [data, data]
                eval_inst_list += [(f"Please type the major texts (ignore small texts on the edge) on {obj} from top to down, "
                                    f"from left to right in the given image. Leave empty if the there is no valid Latin "
                                    f"character in the given image. Ignore small texts in the corner.") for obj in inst['text'].keys()]
                data['human_eval'] = [idx, idx + 1]
                idx += 2
        interface = FreeLabelInterface(
            data_list=data_list,
            eval_inst_list=eval_inst_list
        )
        interface.start()
        for data in self.res_list:
            if FAILED_TOKEN not in data['human_eval']:
                data['human_eval'] = [interface.eval_list[i].lower().strip() for i in data['human_eval']]
            else:
                data['human_eval'] = ['', '']
        self.save()

    def compute_accuracy(self, return_list=False):
        label_list = [[self.normalize_text(t) for t in inst['text'].values()] for inst in self.inst_list]
        model_eval_list = [[self.normalize_text(t) for t in res['model_eval']] for res in self.res_list]
        return self._compute_accuracy(label_list, model_eval_list, return_list)

    def compute_correlation(self):
        label_list = [[self.normalize_text(t) for t in inst['text'].values()] for inst in self.inst_list]
        model_eval_list = [[self.normalize_text(t) for t in res['model_eval']] for res in self.res_list]
        human_eval_list = [[self.normalize_text(t) for t in res['human_eval']] for res in self.res_list]
        return self._compute_correlation(label_list, model_eval_list, human_eval_list)
    

class IOCRGerman(IOCR):
    inst_name = 'i_ocr_german'


class IOCRChinese(IOCR):
    inst_name = 'i_ocr_chinese'
    language = 'chinese'

    def evaluate(self):
        gpt_queries = []
        vlm_queries = []
        idx = 0
        for data in self.res_list:
            if len(data['image_list']) != 1:
                data['model_eval'] = FAILED_TOKEN
                continue
            if self.vlm != 'openai':
                gpt_queries.append(form_openai_mm_query(I_OCR_CHINESE_PROMPT, images=data['image_list']))
            else:
                gpt_queries.append(form_gemini_mm_query(I_OCR_CHINESE_PROMPT, images=data['image_list']))
            vlm_queries.append(form_mm_query(I_OCR_CHINESE_PROMPT, images=data['image_list'], model=self.vlm))
            data['model_eval'] = idx
            idx += 1
        if self.vlm != 'openai':
            gpt_responses = batch(query_openai, gpt_queries, model='chatgpt-4o-latest', temperature=0.0)
        else:
            gpt_responses = batch(query_gemini, gpt_queries, model='gemini-2.5-pro-preview-03-25', temperature=0.0)
        vlm_responses = query_vlm(vlm_queries, model=self.vlm)
        for data in self.res_list:
            if data['model_eval'] != FAILED_TOKEN:
                gpt_res = ''.join(re.findall(r'[\u4e00-\u9fff]', gpt_responses[data['model_eval']]))
                vlm_res = ''.join(re.findall(r'[\u4e00-\u9fff]', vlm_responses[data['model_eval']]))
                data['model_eval'] = [gpt_res, vlm_res]
                data['model_eval_score'] = ''.join(set(data['model_eval'][0]).intersection(set(data['model_eval'][1])))
            else:
                data['model_eval'] = ['', '']
                data['model_eval_score'] = ''
        self.save()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            human_inst_list.append("Please type the major Chinese characters from top to down, from left to right in "
                                   "the given image. Leave empty if the there is no valid Chinese character in the "
                                   "given image. Ignore small texts in the corner.")
            human_res_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = FreeLabelInterface(
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = interface.eval_list[data['human_eval']].lower().strip() if data['human_eval'] != FAILED_TOKEN else ''
        self.save()

    def compute_accuracy(self, return_list=False):
        label_list = [[inst['text'] or FAILED_TOKEN] for inst in self.inst_list]
        model_eval_list = [[res['model_eval_score'] or FAILED_TOKEN] for res in self.res_list]
        return self._compute_accuracy(label_list, model_eval_list, return_list)

    def compute_correlation(self):
        label_list = [[inst['text'] or FAILED_TOKEN] for inst in self.inst_list]
        model_eval_list = [[res['model_eval_score'] or FAILED_TOKEN] for res in self.res_list]
        human_eval_list = [[res['human_eval'] or FAILED_TOKEN] for res in self.res_list]
        return self._compute_correlation(label_list, model_eval_list, human_eval_list)


class IOCRMultiLingual(EvalUnit):
    def __init__(self, model_name: str, sample_size=4):
        self.chinese = IOCRChinese(model_name=model_name, sample_size=sample_size)
        self.german = IOCRGerman(model_name=model_name, sample_size=sample_size)

    @property
    def res_list(self):
        return self.chinese.res_list + self.german.res_list

    def evaluate(self):
        self.chinese.evaluate()
        self.german.evaluate()

    def human_evaluate(self):
        self.chinese.human_evaluate()
        self.german.human_evaluate()

    def compute_accuracy(self, return_list=False):
        chinese_acc = self.chinese.compute_accuracy(return_list)
        germany_acc = self.german.compute_accuracy(return_list)
        if return_list:
            return chinese_acc + germany_acc
        return (chinese_acc + germany_acc) / 2.0

    def compute_correlation(self):
        chines_cor = self.chinese.compute_correlation()
        german_cor = self.german.compute_correlation()
        return (chines_cor[0] + german_cor[0]) / 2.0, (chines_cor[1] + german_cor[1]) / 2.0

    def save(self, save_all=False):
        self.chinese.save(save_all=save_all)
        self.german.save(save_all=save_all)


class IFormatBackground(EvalUnit):
    inst_name = 'i_format_background'
    direction_map = {
        'left half': (0.0, 0.0, 0.5, 1.0),
        'right half': (0.5, 0.0, 1.0, 1.0),
        'upper half': (0.0, 0.0, 1.0, 0.5),
        'lower half': (0.0, 0.5, 1.0, 1.0),
        'left third': (0.0, 0.0, 0.33, 1.0),
        'right third': (0.67, 0.0, 1.0, 1.0),
        'upper third': (0.0, 0.0, 1.0, 0.33),
        'lower third': (0.0, 0.67, 1.0, 1.0),
        'left quarter': (0.0, 0.0, 0.25, 1.0),
        'right quarter': (0.75, 0.0, 1.0, 1.0),
        'upper quarter': (0.0, 0.0, 1.0, 0.25),
        'lower quarter': (0.0, 0.75, 1.0, 1.0),
    }
    complement_map = {
        'left half': (0.5, 0.0, 1.0, 1.0),
        'right half': (0.0, 0.0, 0.5, 1.0),
        'upper half': (0.0, 0.5, 1.0, 1.0),
        'lower half': (0.0, 0.0, 1.0, 0.5),
        'left third': (0.33, 0.0, 0.67, 1.0),
        'right third': (0.33, 0.0, 0.67, 1.0),
        'upper third': (0.0, 0.33, 1.0, 0.67),
        'lower third': (0.0, 0.33, 1.0, 0.67),
        'left quarter': (0.25, 0.0, 0.5, 1.0),
        'right quarter': (0.5, 0.0, 0.75, 1.0),
        'upper quarter': (0.0, 0.25, 1.0, 0.5),
        'lower quarter': (0.0, 0.5, 1.0, 0.75),
    }

    def evaluate(self):
        for idx, (data, inst) in enumerate(zip(self.res_list, self.inst_list)):
            if len(data['image_list']) != 1:
                data['auto_eval'] = 0.0
                continue
            width, height = data['image_list'][0].size
            crop_area = self.direction_map[inst['region']]
            crop_area = (
                round(crop_area[0] * width),
                round(crop_area[1] * height),
                round(crop_area[2] * width),
                round(crop_area[3] * height)
            )
            cropped_image = data['image_list'][0].crop(crop_area)
            data['auto_eval'], avg_color = color_condition(cropped_image, inst['color'])

            crop_area = self.complement_map[inst['region']]
            crop_area = (
                round(crop_area[0] * width),
                round(crop_area[1] * height),
                round(crop_area[2] * width),
                round(crop_area[3] * height)
            )
            complement_image = data['image_list'][0].crop(crop_area)
            penalty = count_pixels(complement_image, avg_color)
            data['auto_eval'] = max(0.0, data['auto_eval'] - penalty)
        self.save()

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


class IFormatBorder(IFormatBackground):
    inst_name = 'i_format_border'

    @staticmethod
    def get_border(image):
        width, height = image.size
        min_width = min(width, height)
        border_width = round(width * 0.1)
        border_height = round(height * 0.1)
        top_border = image.crop(((width - min_width) // 2, 0, (width + min_width) // 2, border_height))
        bottom_border = image.crop(((width - min_width) // 2, height - border_height, (width + min_width) // 2, height))
        left_border = image.crop((0, (height - min_width) // 2, border_width, (height + min_width) // 2))
        right_border = image.crop((width - border_width, (height - min_width) // 2, width, (height + min_width) // 2))
        left_border = left_border.rotate(90, expand=True)
        right_border = right_border.rotate(90, expand=True)
        total_height = top_border.height + bottom_border.height + left_border.height + right_border.height
        new_image = Image.new('RGB', (min_width, total_height), (255, 255, 255))
        y_offset = 0
        new_image.paste(top_border, (0, y_offset))
        y_offset += top_border.height
        new_image.paste(left_border, (0, y_offset))
        y_offset += left_border.height
        new_image.paste(right_border, (0, y_offset))
        y_offset += right_border.height
        new_image.paste(bottom_border, (0, y_offset))
        return new_image

    def evaluate(self):
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['auto_eval'] = 0.0
                continue
            border = self.get_border(data['image_list'][0])
            data['auto_eval'], avg_color = color_condition(border, inst['color'])

            width, height = data['image_list'][0].size
            complement_image = data['image_list'][0].crop((
                round(0.1 * width),
                round(0.1 * height),
                round(0.9 * width),
                round(0.9 * height)
            ))
            border = self.get_border(complement_image)
            penalty = count_pixels(border, avg_color)
            data['auto_eval'] = max(0.0, data['auto_eval'] - penalty)
        self.save()


class IEdit(EvalUnit):
    inst_name = 'i_edit'

    def evaluate(self):
        self.load_inst_mm()
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['auto_eval'] = 0.0
                continue
            origin_image = inst['image_list'][0].convert('RGB')
            image = data['image_list'][0].resize(origin_image.size)
            width_margin, height_margin = origin_image.size[0] // 10, origin_image.size[1] // 10
            bbox = (max(inst['bbox'][0] - width_margin, 0),
                    max(inst['bbox'][1] - height_margin, 0),
                    min(inst['bbox'][2] + width_margin, origin_image.size[0]),
                    min(inst['bbox'][3] + height_margin, origin_image.size[1])
                    )
            data['image_list'][0] = image.crop(bbox)

            origin_arr = np.array(origin_image)
            arr = np.array(image)
            origin_arr[bbox[1]: bbox[3], bbox[0]: bbox[2]] = [0, 0, 0]
            arr[bbox[1]: bbox[3], bbox[0]: bbox[2]] = [0, 0, 0]
            data['auto_eval'] = calculate_ssim(arr, origin_arr)

        self.save()
        super().evaluate()

    def human_evaluate(self):
        self.load_inst_mm()
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                continue
            origin_image = inst['image_list'][0].convert('RGB')
            image = data['image_list'][0].resize(origin_image.size)
            width_margin, height_margin = origin_image.size[0] // 10, origin_image.size[1] // 10
            bbox = (max(inst['bbox'][0] - width_margin, 0),
                    max(inst['bbox'][1] - height_margin, 0),
                    min(inst['bbox'][2] + width_margin, origin_image.size[0]),
                    min(inst['bbox'][3] + height_margin, origin_image.size[1])
                    )
            data['image_list'][0] = image.crop(bbox)
        super().human_evaluate()

    def compute_accuracy(self, return_list=False):
        model_eval_list = super().compute_accuracy(return_list=True)
        auto_eval_list = [res['auto_eval'] for res in self.res_list]

        combined_list = [a * m for a, m in zip(auto_eval_list, model_eval_list)]
        if return_list:
            return combined_list
        return np.mean(combined_list)

    def compute_correlation(self):
        return super().compute_correlation()


class IEditText(IEdit, IOCR):
    inst_name = 'i_edit_text'


class IEditObjectAdd(IEdit, IObjectInclude):
    inst_name = 'i_edit_object_add'


class IEditObjectRemove(IEdit, IObjectExclude):
    inst_name = 'i_edit_object_remove'


class IEditObjectModify(IEdit, IObjectInclude):
    inst_name = 'i_edit_object_modify'
    vlm = 'gemini'


class IEditAdd(EvalUnit):
    inst_name = 'i_edit_add'

    def evaluate(self):
        self.load_inst_mm()
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['image_list']) != 1:
                data['auto_eval'] = [0.0, 0.0]
                continue
            origin_image = inst['ref_image_list'][0].convert('RGB')
            image = data['image_list'][0].resize(origin_image.size)
            width_margin, height_margin = origin_image.size[0] // 10, origin_image.size[1] // 10
            bbox = (max(inst['bbox'][0] - width_margin, 0),
                    max(inst['bbox'][1] - height_margin, 0),
                    min(inst['bbox'][2] + width_margin, origin_image.size[0]),
                    min(inst['bbox'][3] + height_margin, origin_image.size[1])
                    )
            cropped_image = image.crop(bbox)
            cropped_origin_image = origin_image.crop(bbox)
            origin_arr = np.array(origin_image)
            arr = np.array(image)
            origin_arr[bbox[1]: bbox[3], bbox[0]: bbox[2]] = [0, 0, 0]
            arr[bbox[1]: bbox[3], bbox[0]: bbox[2]] = [0, 0, 0]
            data['auto_eval'] = [calculate_ssim(arr, origin_arr), calculate_dreamsim(cropped_image, cropped_origin_image)]
        self.save()

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [res['auto_eval'][0] * res['auto_eval'][1] for res in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


class IEditColor(IEditAdd):
    inst_name = 'i_edit_color'


if __name__ == '__main__':
    pass
