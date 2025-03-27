import unicodedata
import string

from eval import *
from prompt import *


class IObject(EvalUnit):
    label_list: tuple

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
            queries += [form_openai_mm_query(
                IMAGE_TOKEN(0) + self.instruction_func(obj),
                images=data['image_list']) for obj in obj_list
            ]
            data['model_eval'] = list(range(idx, idx + len(obj_list)))
            idx += len(obj_list)
        responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.0)
        parsed_responses = [self.gpt_judge_process_func(res) for res in responses]
        for data in self.res_list:
            data['model_eval'] = [parsed_responses[idx] for idx in data['model_eval']]
        self.save()

        if self.inst_name == 'i_object_counting':
            for data, inst in zip(self.res_list, self.inst_list):
                data['model_eval'] = float(data['model_eval'] == (inst['count'] - 2))

    def human_eval(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            obj_list = [inst['object']] if isinstance(inst['object'], str) else inst['object']
            human_inst_list += [self.human_instruction_func(obj) for obj in obj_list]
            human_res_list += [data] * len(obj_list)
            data['human_eval'] = list(range(idx, idx + len(obj_list)))
            idx += len(obj_list)
        interface = MultiLabelInterface(
            label_list=self.label_list,
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = [self.human_judge_process_func(interface.eval_list[idx]) for idx in data['human_eval']]
        self.save()

        if self.inst_name == 'i_object_counting':
            for data, inst in zip(self.res_list, self.inst_list):
                data['human_eval'] = float(data['human_eval'] == (inst['count'] - 2))
            self.save()

    def calculate_metrics(self):
        model_eval_list = [np.mean(res['model_eval']) for res in self.res_list]
        human_eval_list = [np.mean(res['human_eval']) for res in self.res_list]
        print(f"GPT evaluation accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        model_eval_list = np.concatenate([res['model_eval'] for res in self.res_list])
        human_eval_list = np.concatenate([res['human_eval'] for res in self.res_list])
        print(f"Cohen's Kappa for {self.inst_name}: ", calculate_kappa(model_eval_list, human_eval_list))
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))


class IObjectInclude(IObject):
    inst_name = 'i_object_include'
    label_list = ('Yes', 'No')

    @staticmethod
    def instruction_func(obj):
        return I_OBJECT_EXIST_PROMPT(obj)

    @staticmethod
    def human_instruction_func(obj):
        return f"Is/Are there {obj} in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if res.lower().startswith('yes') else 0.0

    @staticmethod
    def human_judge_process_func(res: str):
        return 1.0 if res == 0 else 0.0


class IObjectAttribute(IObjectInclude):
    inst_name = 'i_object_attribute'
    

class IObjectExclude(IObjectInclude):
    inst_name = 'i_object_exclude'

    @staticmethod
    def gpt_judge_process_func(res: str):
        return 1.0 if res.lower().startswith('no') else 0.0

    @staticmethod
    def human_judge_process_func(res: str):
        return 1.0 if res == 1 else 0.0


class IObjectCoT(IObjectInclude):
    inst_name = 'i_object_cot'
    label_list = ("Yes", "No")

    @staticmethod
    def instruction_func(obj):
        return f"Is the given image about {obj}? Answer only yes or no.\n"

    @staticmethod
    def human_instruction_func(obj):
        return f"Is the given image about {obj}?\n"


class IObjectCounting(IObject):
    inst_name = 'i_object_counting'
    label_list = ("A. Less than 3", "B. 3", "C. 4", "D. 5", "E. 6", "F. More than 6")

    @staticmethod
    def instruction_func(obj):
        return I_OBJECT_COUNT_PROMPT(obj)

    @staticmethod
    def human_instruction_func(obj):
        return f"How many {obj} are there in the given image?\n"

    @staticmethod
    def gpt_judge_process_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd', 'e', 'f') else FAILED_TOKEN

    @staticmethod
    def human_judge_process_func(res: str):
        return res


class ISpacial(EvalUnit):
    @staticmethod
    @abstractmethod
    def instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def human_instruction_func(inst: dict):
        pass

    @staticmethod
    @abstractmethod
    def gpt_judge_parse_func(res: str):
        pass

    @staticmethod
    @abstractmethod
    def gpt_judge_process_func(res: str, const: tuple):
        pass

    @staticmethod
    @abstractmethod
    def human_judge_process_func(res: str):
        pass

    def evaluate(self):
        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            for constraint in inst['constraints']:
                queries.append(form_openai_mm_query(
                    IMAGE_TOKEN(0) + self.instruction_func(constraint),
                    images=data['image_list']
                ))
            data['model_eval'] = list(range(idx, idx + len(inst['constraints'])))
            idx += len(inst['constraints'])
        responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.0)
        parsed_responses = [self.gpt_judge_parse_func(res) for res in responses]
        for data, inst in zip(self.res_list, self.inst_list):
            data['model_eval'] = [self.gpt_judge_process_func(parsed_responses[model_eval], constraint) for
                                model_eval, constraint in zip(data['model_eval'], inst['constraints'])]
        self.save()

    def human_evaluate(self):
        human_queries = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            human_res_list += [data] * len(inst['constraints'])
            for constraint in inst['constraints']:
                human_queries.append(self.human_instruction_func(constraint))
            data['human_eval'] = list(range(idx, idx + len(inst['constraints'])))
            idx += len(inst['constraints'])
        interface = MultiLabelInterface(
            label_list=("Yes", "No"),
            eval_inst_list=human_queries,
            data_list=human_res_list
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = [self.human_judge_process_func(interface.eval_list[idx]) for idx in data['human_eval']]
        self.save()

    def calculate_metrics(self):
        model_eval_list = [np.mean(res['model_eval']) for res in self.res_list]
        human_eval_list = [np.mean(res['human_eval']) for res in self.res_list]
        print(f"Auto evaluation accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        model_eval_list = np.concatenate([res['model_eval'] for res in self.res_list])
        human_eval_list = np.concatenate([res['human_eval'] for res in self.res_list])
        print(f"Cohen's Kappa for {self.inst_name}: ", calculate_kappa(model_eval_list, human_eval_list))
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))


class ISpacialAbsolute(ISpacial):
    inst_name = 'i_spacial_absolute'

    @staticmethod
    def instruction_func(const: dict):
        return I_SPACIAL_ABSOLUTE_PROMPT(const[0])

    @staticmethod
    def human_instruction_func(const: dict):
        return f'Is there {const[0]} in the given image and locate at the {const[1]} part of the image?\n'

    @staticmethod
    def gpt_judge_parse_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd', 'e', 'f') else FAILED_TOKEN

    @staticmethod
    def gpt_judge_process_func(res: str, const: tuple):
        return float(res == {'bottom left': 0, 'bottom right': 1, 'top left': 2, 'top right': 3}[const[1]])

    @staticmethod
    def human_judge_process_func(res: str):
        return float(res == 0)


class ISpacialRelative(ISpacial):
    inst_name = 'i_spacial_relative'

    @staticmethod
    def instruction_func(const: dict):
        if const[2] in ('to the left of', 'to the right of'):
            return I_SPACIAL_RELATIVE_FR(const[0], const[1])
        else:
            return I_SPACIAL_RELATIVE_UD(const[0], const[1])

    @staticmethod
    def human_instruction_func(const: dict):
        return f'Is {const[0]} {const[2]} {const[1]} in the given image?\n'

    @staticmethod
    def gpt_judge_parse_func(res: str):
        return ord(res.lower()[0]) - 97 if res.lower()[0] in ('a', 'b', 'c', 'd') else FAILED_TOKEN

    @staticmethod
    def gpt_judge_process_func(res: str, const: tuple):
        if const[2] in ('to the left of', 'above'):
            return float(res == 0)
        else:
            return float(res == 1)

    @staticmethod
    def human_judge_process_func(res: str):
        return float(res == 0)


class IOCR(EvalUnit):
    inst_name = 'i_ocr'
    language = 'english'

    def evaluate(self):
        queries = []
        for data in self.res_list:
            queries.append(form_openai_mm_query(
                IMAGE_TOKEN(0) + I_OCR_ENGLISH_PROMPT('the given image'),
                images=data['image_list']
            ))
        responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.0)
        pattern = r'(\[.*\]?)'
        for data, res in zip(self.res_list, responses):
            text_res = re.search(pattern, res)
            if text_res is not None:
                text_res = eval(text_res.group(1))
            else:
                text_res = []
            data['model_eval'] = ' '.join(text_res).lower().strip()
        self.save()

    def human_evaluate(self):
        interface = FreeLabelInterface(
            eval_inst_list=["Please type the major texts (ignore small texts on the edge) from top to down, "
                            "from left to right in the given image. Leave empty if the there is no valid Latin "
                            "character in the given image. Ignore small texts in the corner."] * len(self.res_list),
            data_list=self.res_list
        )
        interface.start()
        for data, human_eval in zip(self.res_list, interface.eval_list):
            data['human_eval'] = human_eval.lower().strip()
        self.save()

    def calculate_metrics(self, ignore_null=True):
        label_list = [[self.normalize_text(inst['text'])] for inst, res in zip(self.inst_list, self.res_list) if res['model_eval'] != '' or not ignore_null]
        model_eval_list = [[self.normalize_text(res['model_eval'])] for res in self.res_list if res['model_eval'] != '' or not ignore_null]
        human_eval_list = [[self.normalize_text(res['human_eval'])] for res in self.res_list if res['model_eval'] != '' or not ignore_null]
        self._calculate_metrics(label_list, model_eval_list, human_eval_list)

    @staticmethod
    def normalize_text(text):
        normalized_text = text.lower().strip()
        normalized_text = normalized_text.translate(str.maketrans('', '', string.punctuation))
        normalized_text = unicodedata.normalize('NFD', normalized_text)
        normalized_text = ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')
        return normalized_text if normalized_text != '' else FAILED_TOKEN

    def _calculate_metrics(self, label_list, model_eval_list, human_eval_list):
        wer = evaluate.load('wer') if self.language == 'english' else evaluate.load('cer')
        wer_list = [1.0 - wer.compute(predictions=model_eval, references=human_eval)
                    for model_eval, human_eval in zip(model_eval_list, human_eval_list)]
        print(f"Word Error Rate for {self.inst_name}: ", np.mean(wer_list))
        wer_list = [1.0 - wer.compute(predictions=model_eval, references=label)
                    for model_eval, label in zip(model_eval_list, label_list)]
        print(f"GPT evaluation Word Error Rate for {self.inst_name}: ", np.mean(wer_list))
        wer_list = [1.0 - wer.compute(predictions=human_eval, references=label)
                    for human_eval, label in zip(human_eval_list, label_list)]
        print(f"Human evaluation Word Error Rate for {self.inst_name}: ", np.mean(wer_list))


class IOCRTwo(IOCR):
    inst_name = 'i_ocr_two'

    def evaluate(self):
        queries = []
        for data, inst in zip(self.res_list, self.inst_list):
            queries.append(form_openai_mm_query(
                IMAGE_TOKEN(0) + I_OBJECT_EXIST_PROMPT(inst['object']),
                images=data['image_list']
            ))
        responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.0)
        for data, res in zip(self.res_list, responses):
            if res.lower().startswith('yes'):
                data['model_eval'] = [0.0, 0.0]
            else:
                data['model_eval'] = [FAILED_TOKEN, FAILED_TOKEN]
        self.save()

        queries = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if FAILED_TOKEN not in data['model_eval']:
                queries += [form_openai_mm_query(
                    IMAGE_TOKEN(0) + I_OCR_ENGLISH_PROMPT('ONLY' + obj),
                    images=data['image_list']) for obj in inst['text'].keys()
                ]
                data['model_eval'] = [idx, idx + 1]
                idx += 2
        responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.0)
        pattern = r'(\[.*\]?)'
        for data in self.res_list:
            if FAILED_TOKEN not in data['model_eval']:
                for i in range(2):
                    text_res = re.search(pattern, responses[data['model_eval'][i]])
                    if text_res is not None:
                        text_res = eval(text_res.group(1))
                    else:
                        text_res = []
                    data['model_eval'][i] = ' '.join(text_res).lower().strip()
        self.save()

    def human_evaluate(self):
        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=[f"Is/Are there {inst['object']} in the given image?" for inst in self.inst_list],
            data_list=self.res_list
        )
        interface.start()
        for data, human_eval in zip(self.res_list, interface.eval_list):
            if human_eval == 0:
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
        self.save()

    def calculate_metrics(self):
        label_list = [[self.normalize_text(t) for t in inst['text'].values()] for inst in self.inst_list]
        model_eval_list = [[self.normalize_text(t) for t in res['model_eval']] for res in self.res_list]
        human_eval_list = [[self.normalize_text(t) for t in res['human_eval']] for res in self.res_list]
        self._calculate_metrics(label_list, model_eval_list, human_eval_list)
    

class IOCRGerman(IOCR):
    inst_name = 'i_ocr_german'


class IOCRChinese(IOCR):
    inst_name = 'i_ocr_chinese'
    language = 'chinese'

    def evaluate(self):
        queries = []
        for data in self.res_list:
            queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + I_OCR_CHINESE_PROMPT, images=data['image_list']))
        responses = batch(query_openai, queries, model='chatgpt-4o-latest', temperature=0.0)
        for data, res in zip(self.res_list, responses):
            data['model_eval'] = [''.join(re.findall(r'[\u4e00-\u9fff]', res))]
        self.save()

        queries = []
        for data in self.res_list:
            queries.append(form_openai_mm_query(IMAGE_TOKEN(0) + I_OCR_CHINESE_PROMPT, images=data['image_list']))
        responses = batch(query_openai, queries, model='gemini-2.0-flash-exp', temperature=0.0, dtype='gemini')
        for data, res in zip(self.res_list, responses):
            data['model_eval'].append(''.join(re.findall(r'[\u4e00-\u9fff]', res)))
            data['model_eval_score'] = ''.join(set(res['model_eval'][0]).intersection(set(res['model_eval'][1])))
        self.save()

    def human_evaluate(self):
        interface = FreeLabelInterface(
            eval_inst_list=["Please type the major Chinese characters from top to down, from left to right in the given "
                            "image. Leave empty if the there is no valid Chinese character in the given image. "
                            "Ignore small texts in the corner."] * len(self.res_list),
            data_list=self.res_list
        )
        interface.start()
        for data, human_eval in zip(self.res_list, interface.eval_list):
            data['human_eval'] = human_eval.lower().strip()
        self.save()

    def calculate_metrics(self):
        label_list = [inst['text'] for inst in self.inst_list]
        model_eval_list = [res['model_eval_score'] for res in self.res_list]
        human_eval_list = [res['human_eval'] for res in self.res_list]

        self._calculate_metrics(label_list, model_eval_list, human_eval_list)


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

    def evaluate(self):
        for data, inst in zip(self.res_list, self.inst_list):
            width, height = data['image_list'][0].size
            crop_area = self.direction_map[inst['region']]
            crop_area = (
                round(crop_area[0] * width),
                round(crop_area[1] * height),
                round(crop_area[2] * width),
                round(crop_area[3] * height)
            )
            cropped_image = data['image_list'][0].crop(crop_area)
            data['auto_eval'] = color_condition(cropped_image, inst['color'])
        self.save()

    def calculate_metrics(self):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy (SSIM) for {self.inst_name}: ", np.mean(auto_eval_list))


class IFormatSymmetric(EvalUnit):
    inst_name = 'i_format_symmetric'

    def evaluate(self):
        for data, inst in zip(self.res_list, self.inst_list):
            data['auto_eval'] = symmetry_condition(data['image_list'][0], inst['symmetry_type'])
        self.save()

    def calculate_metrics(self):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy (SSIM) for {self.inst_name}: ", np.mean(auto_eval_list))


class IEdit(EvalUnit):
    inst_name = 'i_edit'

    def evaluate(self):
        self.load_inst_mm()
        for data, inst in zip(self.res_list, self.inst_list):
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

    def calculate_metrics(self):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy (SSIM) for {self.inst_name}: ", np.mean(auto_eval_list))
        super().calculate_metrics()


class IEditText(IEdit, IOCR):
    inst_name = 'i_edit_text'

    def calculate_metrics(self):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy (SSIM) for {self.inst_name}: ", np.mean(auto_eval_list))
        IOCR.calculate_metrics(self, ignore_null=False)


class IEditObjectAdd(IEdit, IObjectInclude):
    inst_name = 'i_edit_object_add'


class IEditObjectRemove(IEdit, IObjectExclude):
    inst_name = 'i_edit_object_remove'


class IEditObjectModify(IEdit, IObjectInclude):
    inst_name = 'i_edit_object_modify'


if __name__ == '__main__':
    a = IOCRTwo(model_name='Dalle3', sample_size=2)
    a.calculate_metrics()
