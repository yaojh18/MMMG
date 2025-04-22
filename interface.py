import threading
import random
import time

import io
import sys
from abc import abstractmethod

from utils import *


class Interface:
    def __init__(self, data_list):
        self.data_list = data_list
        self.is_finished = threading.Event()
        self.interface = self.construct_interface()

    def start(self):
        thread = threading.Thread(target=self.interface.launch, kwargs={'share': True})
        thread.start()
        self.is_finished.wait()

    @abstractmethod
    def construct_interface(self):
        pass

    def update_interface(self):
        pass


class MultiLabelInterface(Interface):
    def __init__(self, label_list, eval_inst_list, mm_type='i',
                 ref_list=None, back_list=None, shuffle=False, multi_choice=False, **kwargs):
        self.label_list = label_list
        self.eval_inst_list = eval_inst_list
        self.mm_type = mm_type
        self.ref_list = ref_list
        self.back_list = back_list
        self.shuffle = shuffle
        self.multi_choice = multi_choice
        self.eval_list = [FAILED_TOKEN] * len(self.eval_inst_list)
        super().__init__(**kwargs)

    def construct_interface(self):
        if self.shuffle:
            self.idx_list = random.sample(range(len(self.data_list)), len(self.data_list))
        else:
            self.idx_list = list(range(len(self.data_list)))
        self.eval_inst_list = [self.eval_inst_list[i] for i in self.idx_list]
        self.data_list = [self.data_list[i] for i in self.idx_list]
        if self.ref_list is not None:
            self.ref_list = [self.ref_list[i] for i in self.idx_list]
        with (gr.Blocks() as interface):
            if self.back_list is not None:
                for back in self.back_list:
                    with gr.Accordion(back[0] + ' examples', open=False):
                        for audio in back[1:]:
                            gr.Audio(value=audio, type="filepath")
            current_index = gr.State(0)
            if self.mm_type == 'i':
                mm_com = gr.Image(
                    value=self.data_list[0]['image_list'][0],
                    visible=True,
                    label="Response",
                    width=400,
                    height=400
                )
            else:
                mm_com = gr.Audio(
                    value=(SAMPLE_RATE, self.data_list[0]['audio_list'][0]),
                    visible=True,
                    label="Response",
                    type="numpy"
                )
            if self.ref_list is not None:
                if self.mm_type == 'i':
                    mm_ref_com = gr.Image(
                        value=self.ref_list[0],
                        visible=True,
                        label="Reference",
                        width=400,
                        height=400
                    )
                else:
                    mm_ref_com = gr.Audio(
                        value=(SAMPLE_RATE, self.ref_list[0]),
                        visible=True,
                        label="Reference",
                        type="numpy"
                    )
            eval_textbox = gr.Textbox(
                value=self.eval_inst_list[0],
                label="Evaluation",
                interactive=False
            )
            if self.multi_choice:
                judgement_choice = gr.CheckboxGroup(
                    choices=self.label_list,
                    label="Judgement"
                )
            else:
                judgement_choice = gr.Radio(
                    choices=self.label_list,
                    label="Judgement"
                )
            with gr.Row():
                next_button = gr.Button("Next", interactive=False)
                prev_button = gr.Button("Prev", visible=False, interactive=False)
            next_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(1), judgement_choice],
                outputs=[current_index, mm_com, eval_textbox, judgement_choice, prev_button, next_button]
                        + ([] if self.ref_list is None else [mm_ref_com])
            )
            prev_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(-1), judgement_choice],
                outputs=[current_index, mm_com, eval_textbox, judgement_choice, prev_button, next_button]
                        + ([] if self.ref_list is None else [mm_ref_com])
            )

            def update_buttons_state(judgement):
                return (gr.update(interactive=(judgement is not None and len(judgement) > 0)),
                        gr.update(interactive=(judgement is not None and len(judgement) > 0)))

            judgement_choice.change(
                update_buttons_state,
                inputs=[judgement_choice],
                outputs=[prev_button, next_button]
            )
            return interface

    def update_interface(self, current_index, step, judgement):
        if self.multi_choice:
            self.eval_list[current_index] = [self.label_list.index(j) for j in judgement]
        else:
            self.eval_list[current_index] = self.label_list.index(judgement)
        current_index += step
        if current_index == len(self.data_list):
            self.eval_list = [self.eval_list[self.idx_list.index(i)] for i in range(len(self.eval_inst_list))]
            self.is_finished.set()
            current_index -= 1

        return [
            current_index,
            self.data_list[current_index]['image_list'][0] if self.mm_type == 'i' else (SAMPLE_RATE, self.data_list[current_index]['audio_list'][0]),
            self.eval_inst_list[current_index],
            None,
            gr.update(visible=(current_index - 1) >= 0),
            gr.update(visible=(current_index + 1) <= len(self.data_list))
        ] + ([] if self.ref_list is None else [self.ref_list[current_index] if self.mm_type == 'i' else (SAMPLE_RATE, self.ref_list[current_index])])


class FreeLabelInterface(Interface):
    def __init__(self, eval_inst_list, **kwargs):
        self.eval_inst_list = eval_inst_list
        self.eval_list = [FAILED_TOKEN] * len(self.eval_inst_list)
        super().__init__(**kwargs)

    def construct_interface(self):
        with gr.Blocks() as interface:
            current_index = gr.State(0)
            res_image = gr.Image(
                value=self.data_list[0]['image_list'][0],
                visible=True,
                label="Response",
                width=400,
                height=400
            )
            eval_textbox = gr.Textbox(
                value=self.eval_inst_list[0],
                label="Evaluation",
                interactive=False
            )
            judgement_input = gr.Textbox(
                label="Judgement",
                placeholder="Enter your judgement here."
            )
            with gr.Row():
                next_button = gr.Button("Next")
                prev_button = gr.Button("Prev", visible=False)

            next_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(1), judgement_input],
                outputs=[current_index, res_image, eval_textbox, judgement_input, prev_button, next_button]
            )
            prev_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(-1), judgement_input],
                outputs=[current_index, res_image, eval_textbox, judgement_input, prev_button, next_button]
            )
            return interface

    def update_interface(self, current_index, step, judgement):
        self.eval_list[current_index] = judgement.strip()
        current_index += step
        if current_index == len(self.data_list):
            self.is_finished.set()
            current_index -= 1

        return (
            current_index,
            self.data_list[current_index]['image_list'][0],
            self.eval_inst_list[current_index],
            "",
            gr.update(visible=(current_index - 1) >= 0),
            gr.update(visible=(current_index + 1) <= len(self.data_list))
        )


class LabelBBoxInterface(Interface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.res_list = [FAILED_TOKEN] * len(self.data_list)

    def construct_interface(self):
        import gradio as gr
        from gradio_image_prompter import ImagePrompter
        from gradio_image_prompter.image_prompter import PromptValue
        with gr.Blocks() as interface:
            current_index = gr.State(0)
            inst_textbox = gr.Textbox(
                value=self.data_list[0]['instruction'],
                label="Instruction",
                interactive=False
            )
            interactive_image = ImagePrompter(
                show_label=False,
                value=PromptValue(image=self.data_list[0]['image'], points=[]),
                interactive=True,
                width=600,
                height=600
            )
            bbox_info = gr.Textbox(
                value='',
                label="Bounding Box",
                interactive=False
            )
            with gr.Row():
                next_button = gr.Button("Next")
                prev_button = gr.Button("Prev", visible=False)
                display_button = gr.Button("Display")

            def update_bbox_info(image_prompter):
                points = image_prompter["points"]
                if len(points) > 0:
                    return str({'x1': points[-1][0], 'y1': points[-1][1], 'x2': points[-1][3], 'y2': points[-1][4]})
                else:
                    return ''

            display_button.click(update_bbox_info, inputs=[interactive_image], outputs=[bbox_info])

            next_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(1), interactive_image],
                outputs=[current_index, inst_textbox, interactive_image, bbox_info, prev_button, next_button]
            )
            prev_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(-1), interactive_image],
                outputs=[current_index, inst_textbox, interactive_image, bbox_info, prev_button, next_button]
            )
            return interface

    def update_interface(self, current_index, step, image_prompter):
        from gradio_image_prompter.image_prompter import PromptValue
        import gradio as gr
        points = image_prompter["points"]
        self.res_list[current_index] = (int(points[-1][0]), int(points[-1][1]), int(points[-1][3]), int(points[-1][4])) if len(points) > 0 else None
        current_index += step
        if current_index == len(self.data_list):
            self.is_finished.set()
            current_index -= 1

        return (
            current_index,
            self.data_list[current_index]['instruction'],
            PromptValue(image=self.data_list[current_index]['image'], points=[]),
            '',
            gr.update(visible=(current_index - 1) >= 0),
            gr.update(visible=(current_index + 1) <= len(self.data_list))
        )


class CalibratedLabelInterface(Interface):
    def __init__(self, label_list, eval_inst, **kwargs):
        self.label_list = label_list
        self.eval_inst = eval_inst
        super().__init__(**kwargs)
        self.eval_list = [FAILED_TOKEN] * len(self.data_list)

    def construct_interface(self):
        with gr.Blocks() as interface:
            audios = []
            radios = []
            instruction = gr.Textbox(label='Instruction', value=self.eval_inst)
            for idx, data in enumerate(self.data_list):
                with gr.Row():
                    audios.append(gr.Audio(
                        value=(SAMPLE_RATE, data),
                        label=f'audio_{idx}',
                        type="numpy"
                    ))
                    radio = gr.Radio(
                        choices=self.label_list,
                        label=f'radio_{idx}'
                    )
                    radio.change(self.label, inputs=[gr.State(idx), radio], outputs=[])
                    radios.append(radio)
            message = gr.Textbox(label='Message', value='Successfully submitted!', visible=False)
            submit_button = gr.Button("Submit")
            submit_button.click(self.submit, inputs=[], outputs=[message])
            return interface

    def label(self, i, j):
        self.eval_list[i] = self.label_list.index(j)

    def submit(self):
        self.is_finished.set()
        return gr.update(visible=True)


class StdOutDisplayer:
    def __init__(self):
        self.stdout_capture = None
        self.original_stdout = sys.stdout

    def capture_stdout(self):
        self.interface = self.create_interface()
        thread = threading.Thread(target=self.interface.launch, kwargs={'share': True})
        thread.start()
        time.sleep(10.0)
        self.stdout_capture = io.StringIO()
        sys.stdout = self.stdout_capture

    def get_stdout(self):
        if self.stdout_capture is None:
            return None
        return self.stdout_capture.getvalue()

    def create_interface(self):
        with gr.Blocks() as app:
            output_text = gr.Textbox(
                label="Annotation task URLs (note there might be more than one URL for one task.)",
                interactive=False
            )
            def update_output():
                while True:
                    output = self.get_stdout()
                    if output:
                        marker = 'Human evaluation finished!\n'
                        if marker in output:
                            self.stdout_capture.seek(0)
                            self.stdout_capture.truncate(0)
                            start_pos = output.find(marker) + len(marker)
                            self.stdout_capture.write(output[start_pos:])
                            return output[start_pos:]
                        return output
                    time.sleep(1.0)
            app.load(
                fn=update_output,
                inputs=[],
                outputs=[output_text],
                every=1.0
            )
        return app

    def release_stdout(self):
        if self.stdout_capture:
            sys.stdout = self.original_stdout
            self.stdout_capture = None
            self.interface.close()
