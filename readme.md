# MMMG Benchmark

[**🏆 Leaderboard**](https://yaojh18.github.io/mmmg-leaderboard/#leaderboard) | [**🤗 MMMG**](https://huggingface.co/datasets/UW-FMRL2/MMMG) | [**📖 Paper**]()

This repo contains the evaluation pipeline for the paper "[MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark]()".

[//]: # (## 🔔News)

[//]: # ()
[//]: # (- **🔥[2024-09-05] Introducing [MMMU-Pro]&#40;https://arxiv.org/abs/2409.02813&#41;, a robust version of MMMU benchmark for multimodal AI evaluation! 🚀**)

[//]: # (- **🚀[2024-01-31]: We added Human Expert performance on the [Leaderboard]&#40;https://mmmu-benchmark.github.io/#leaderboard&#41;!🌟**)

[//]: # (- **🔥[2023-12-04]: Our evaluation server for test set is now availble on [EvalAI]&#40;https://eval.ai/web/challenges/challenge-page/2179/overview&#41;. We welcome all submissions and look forward to your participation! 😆**)

## Introduction

Automatically evaluating multimodal generation presents a significant challenge, as automated metrics often struggle to align reliably with human evaluation, especially for complex tasks that involve multiple modalities. To address this, we present MMMG, a comprehensive and human-aligned benchmark for multimodal generation across 4 modality combinations (image, audio, interleaved text and image, interleaved text and audio), with a focus on tasks that present significant challenges for generation models, while still enabling reliable automatic evaluation through a combination of models and programs. MMMG encompasses 49 tasks (including 29 newly developed ones), each with a carefully designed evaluation pipeline, and 937 instructions to systematically assess reasoning, controllability, and other key capabilities of multimodal generation models. Extensive validation demonstrates that MMMG is highly aligned with human evaluation, achieving an average agreement of 94.3%. Benchmarking results on 24 multimodal generation models reveal that even though the state-of-the-art model, GPT Image, achieves 78.3% accuracy for image generation, it falls short on multimodal reasoning and interleaved generation. Furthermore, results suggest considerable headroom for improvement in audio generation, highlighting an important direction for future research.

![Alt text](assets/main.png)

## Evaluation
### Generate Responses in Required Format for Parsing
In the generation stage, MMMG requires only torch of any version and sound pakages (soundfile and librosa) that is usually compatible with your model environment. 
```bash
conda activate your_env
pip install -r requirements_light.txt --upgrade-strategy only-if-needed
# if you have a compatible environment (check requirements.txt), you can have a single conda evaluation environment for both generation and evaluation by running the following command in your model's environment:
# pip install -r requirements.txt --upgrade-strategy only-if-needed
```
After that, implement the `generate` function for your model in `model_customized.py`. Make sure you strictly follow the format requirement specified in `model_customized.py`. Run the following instruction to generate all responses for all task:
```bash
python eval_pipeline.py --model_name model_name --category category --job generate
# model_name is the same name as your implemented model class name in model_customized.py
# category can be one of i, it, a, at, representing image, interleaved image-text, sound + music and speech + interleaved speech-text generation.
```
You should see a `./output/{model_name}/` folder under the root dir, which stores the generated responses.
### Evaluate Generate Responses
To test your model on MMMG, you should apply for [OpenAI API key](https://platform.openai.com/api-keys) and [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) and then run
```bash
# python >= 3.9 is required
conda create --name mmmg python=3.9
conda activate mmmg
pip install -r requirements.txt
export OPENAI_KEY=openai_key # change to you OpenAI API key
export GEMINI_KEY=gemini_key # change to you Gemini API key
python eval_pipeline.py --model_name model_name --category category  --job evaluate
```
You can also manually add your API keys at Line 22-23 in `utils.py` to permanently store the API keys.

You should see a `./output/{model_name}/{category}.csv` file, which stores the evaluation scores of your model. To submit your model's scores to leaderboard, please refer to [leaderboard](https://yaojh18.github.io/mmmg-leaderboard/).

## Baseline Models
We provide the implementation of all baselines in `model.py`, `model_image.py`, `model_audio.py` and `model_interleaved.py`. Your can use the implemented model class name for evaluation directly. To run these baselines models, please first download all the models files from [Google drive link](https://drive.google.com/drive/folders/1-szZ4c1kSBhONYgeiUbCgKLGjhHD_lxi?usp=drive_link) and place them under the root dir, your file structure should look like this:
```aiignore
root/
├── models/
│   ├── Anole/
│   ├── Seed/
│   └── ...
```
Then setup model-specific environment by the `setup.sh` file under each model folder. Environmental configs of models without a corresponding model folder are in `./models/Others/setup.sh` and make sure you pass the correct API keys. To access the our evaluation results of baseline models, please download from [Google drive link](https://drive.google.com/drive/folders/183cvq-4Rz0NaWf3X6VWpr7vdbwjx6hf0?usp=drive_link).
## Human Evaluation
To replicate the human evaluation pipeline reported in paper, please run:
```bash
pip install gradio
python eval_pipeline.py --model_name model_name --category category --job human
```

## Contact
- Jihan Yao: jihany2@cs.washington.edu
- Yushi Hu: yushihu@uw.edu

## Citation

**BibTeX:**
```bibtex
comming soon
```