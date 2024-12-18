import threading
import gradio as gr
from abc import abstractmethod

from utils import *


class Interface:
    def __init__(self, data_list):
        self.data_list = data_list
        self.is_finished = threading.Event()
        self.interface = self.construct_interface()

    def start(self):
        thread = threading.Thread(target=self.interface.launch, kwargs={'server_port': 7960})
        thread.start()
        self.is_finished.wait()

    @abstractmethod
    def construct_interface(self):
        pass

    @abstractmethod
    def update_interface(self):
        pass


class MultiLabelInterface(Interface):
    def __init__(self, label_list, eval_inst_list, **kwargs):
        self.label_list = label_list
        self.eval_inst_list = eval_inst_list
        self.eval_list = [FAILED_TOKEN] * len(self.eval_inst_list)
        super().__init__(**kwargs)

    def construct_interface(self):
        with gr.Blocks() as interface:
            current_index = gr.State(0)
            inst_textbox = gr.Textbox(
                value=self.data_list[0]['query'],
                label="Instruction",
                interactive=False
            )
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
                outputs=[current_index, inst_textbox, res_image, eval_textbox, judgement_choice, prev_button, next_button]
            )
            prev_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(-1), judgement_choice],
                outputs=[current_index, inst_textbox, res_image, eval_textbox, judgement_choice, prev_button, next_button]
            )

            def update_buttons_state(judgement):
                return gr.update(interactive=(judgement is not None)), gr.update(interactive=(judgement is not None))

            judgement_choice.change(
                update_buttons_state,
                inputs=[judgement_choice],
                outputs=[prev_button, next_button]
            )
            return interface

    def update_interface(self, current_index, step, judgement):
        self.eval_list[current_index] = self.label_list.index(judgement)
        current_index += step
        if current_index == len(self.data_list):
            self.is_finished.set()
            current_index -= 1

        return (
            current_index,
            self.data_list[current_index]['query'],
            self.data_list[current_index]['image_list'][0],
            self.eval_inst_list[current_index],
            None,
            gr.update(visible=(current_index - 1) >= 0),
            gr.update(visible=(current_index + 1) <= len(self.data_list))
        )


class FreeLabelInterface(Interface):
    def __init__(self, eval_inst_list, **kwargs):
        self.eval_inst_list = eval_inst_list
        self.eval_list = [None] * len(self.eval_inst_list)
        super().__init__(**kwargs)

    def construct_interface(self):
        with gr.Blocks() as interface:
            current_index = gr.State(0)
            inst_textbox = gr.Textbox(
                value=self.data_list[0]['query'],
                label="Instruction",
                interactive=False
            )
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
                next_button = gr.Button("Next", interactive=False)
                prev_button = gr.Button("Prev", visible=False, interactive=False)

            next_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(1), judgement_input],
                outputs=[current_index, inst_textbox, res_image, eval_textbox, judgement_input, prev_button, next_button]
            )
            prev_button.click(
                self.update_interface,
                inputs=[current_index, gr.State(-1), judgement_input],
                outputs=[current_index, inst_textbox, res_image, eval_textbox, judgement_input, prev_button, next_button]
            )

            def update_buttons_state(judgement):
                return gr.update(interactive=(judgement.strip() != "")), gr.update(interactive=(judgement.strip() != ""))

            judgement_input.change(
                update_buttons_state,
                inputs=[judgement_input],
                outputs=[prev_button, next_button]
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
            self.data_list[current_index]['query'],
            self.data_list[current_index]['image_list'][0],
            self.eval_inst_list[current_index],
            "",
            gr.update(visible=(current_index - 1) >= 0),
            gr.update(visible=(current_index + 1) <= len(self.data_list))
        )
