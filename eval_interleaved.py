from eval import EvalUnit
from prompt import *
from interface import *


class IConsistencySemantic(EvalUnit):
    inst_name = 'i_consistency_semantic'

    def evaluate(self):
        query_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            data['model_eval'] = []
            for image, target in zip(data['image_list'], inst['object']):
                query_list.append(form_openai_mm_query(IMAGE_TOKEN(0) + I_OBJECT_EXIST_PROMPT(target), images=[image]))
                data['model_eval'].append(idx)
                idx += 1
            data['model_eval'] += [FAILED_TOKEN] * (len(inst['object']) - len(data['model_eval']))

        responses = batch(query_openai, query_list, model='chatgpt-4o-latest', temperature=0.0)
        for data in self.res_list:
            data['model_eval'] = [float('yes' in responses[i].lower()) if i != FAILED_TOKEN else 0.0
                                  for i in data['model_eval']]
        self.save()

    def human_evaluate(self):
        human_inst_list = []
        human_res_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            data['human_eval'] = []
            for image, target in zip(data['image_list'], inst['object']):
                human_inst_list.append(f"Is/Are there {target} in the given image?")
                human_res_list.append({"image_list": [image]})
                data['human_eval'].append(idx)
                idx += 1
            data['human_eval'] += [FAILED_TOKEN] * (len(inst['object']) - len(data['human_eval']))
        interface = MultiLabelInterface(
            label_list=("Yes", "No"),
            eval_inst_list=human_inst_list,
            data_list=human_res_list,
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = [1.0 - interface.eval_list[i] if i != FAILED_TOKEN else 0.0
                                  for i in data['human_eval']]
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


if __name__ == '__main__':
    a = IConsistencySemantic(model_name='Dalle3', sample_size=2)
    # a.evaluate()
    # a.calculate_metrics()
