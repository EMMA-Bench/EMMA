<p align="center">
    <img src="assets/emma-small.jpg" width="30%"> <br>
</p>

# EMMA: An Enhanced MultiModal ReAsoning Benchmark

🌟  This is the official repository for the paper "[Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark]()", which contains generation and evaluation code for the **EMMA** benchmark.

[[🌐 Homepage](https://emma-benchmark.github.io/)] [[🤗EMMA](https://huggingface.co/datasets/luckychao/EMMA)] [[🤗EMMA-mini](https://huggingface.co/datasets/luckychao/EMMA-mini)] [[📖 ArXiv Paper](https://www.arxiv.org/abs/2501.05444)]

## 👀 About EMMA

The ability to organically reason **over** and **with** both text and images is a pillar of human intelligence, yet the ability of Multimodal Large Language Models (MLLMs) to perform such multimodal reasoning remains under-explored.
We introduce **EMMA (Enhanced MultiModal reAsoning)**, a benchmark targeting organic multimodal reasoning across mathematics, physics, chemistry, and coding. 
EMMA tasks demand advanced cross-modal reasoning that cannot be solved by thinking separately in each modality, offering an enhanced test suite for MLLMs' reasoning capabilities. 

EMMA is composed of 2,788 problems, of which 1,796 are newly constructed, across four domains. Within each subject, we further provide fine-grained labels for each question based on the specific skills it measures. 

<p align="center">
    <img src="assets/EMMA.jpg" width="90%"> <br>
  <b>Overview of EMMA.</b> 
</p>

Our evaluation of state-of-the-art MLLMs on EMMA reveals significant limitations in handling complex multimodal and multi-step reasoning tasks, with even advanced techniques like Chain-of-Thought prompting and test-time compute scaling underperforming. 
These findings underscore the need for improved multimodal architectures and training paradigms to close the gap between human and model reasoning in multimodality. 

## 🏆 Leaderboard

The leaderboard is available [here](https://emma-benchmark.github.io/#leaderboard).

## 📖 Dataset Usage

### Data Downloading

To create a more balanced subset of EMMA, we randomly sample 400 questions (100 per subject) from the benchmark and get EMMA-mini[🤗]().

You can download both two datasets by the following command (Taking downloading math data as an example):

```python
from datasets import load_dataset

dataset = load_dataset("luckychao/EMMA", "Math", split="test")
```

```python
from datasets import load_dataset

dataset = load_dataset("luckychao/EMMA-mini", "Math", split="test")
```

### Data Format

The dataset is provided in jsonl format and contains the following attributes:

```
{
    "pid": [string] Problem ID, e.g., “math_1”,
    "question": [string] The question text,
    "options": [list] Choice options for multiple-choice problems. For free-form problems, this could be a 'none' value,
    "answer": [string] The correct answer for the problem,
    "image_1": [image] ,
    "image_2": [image] ,
    "image_3": [image] ,
    "image_4": [image] ,
    "image_5": [image] ,
    "solution": [string] The detailed thinking steps required to solve the problem,
    "subject": [string] The subject of data, e.g., “Math”, “Physics”...,
    "task": [string] The task of the problem, e.g., “Code Choose Vis”,
    "category": [string] The category of the problem, e.g., “2D Transformation”,
    "source": [string] The original source dataset of the data, e.g., “math-vista”. For handmade data, this could be “Newly annotated” ,
    "type": [string] Types of questions, e.g., “Multiple Choice”, “Open-ended”,
    "context": [string] Background knowledge required for the question. For problems without context, this could be a 'none' value,
}
```

## 📈 Evaluation

### Responses Generation
Our repository supports the evaluation of open source models such as Qwen2-VL, InternVL, LLaVA, and closed source models such as GPT, Gemini, Claude, etc. 
You can generate responses of these models by using the following commands:

Open-source Model:
```
 python generate_response.py \
 --dataset_name 'mm-reasoning/EMMA' \
 --split 'test' \
 --subject 'Math' 'Physics' 'Chemistry' 'Coding' \
 --strategy 'CoT' \
 --config_path 'configs/gpt.yaml' \
 --model_path 'path_to_your_local_model' \
 --output_path 'path_to_output_file' \
 --max_tokens 4096 \
 --temperature 0.7 \
 --save_every 20
```

Close-source Model:

```
 python generate_response.py \
 --dataset_name 'mm-reasoning/EMMA' \
 --split 'test' \
 --subject 'Math' 'Physics' 'Chemistry' 'Coding' \
 --strategy 'CoT' \
 --config_path 'configs/gpt.yaml' \
 --model 'remote-model-name' \
 --api_key '' \
 --output_path 'path_to_output_file' \
 --max_tokens 4096 \
 --temperature 0.7 \
 --save_every 20
```

### Answer Evaluation

Once all the model outputs have been generated, execute the `evaluate.py` function to extract the short answer text from the detailed response and evaluate the correctness of the answers.
We offer two evaluation methods: **fast-eval** and **LLMs-eval**. The fast-eval method employs rule-based extraction for quicker processing, while the LLMs-eval method leverages advanced models like GPT-4o to enhance precision in extraction and evaluation.

Fast-extract:
```
python evaluate.py \
--results_dir 'path_to_your_results_dir' \
--response_label 'response' \
--save_every 20
```

LLMs-eval:
```
python evaluate.py \
--results_dir 'path_to_your_results_dir' \
--response_label 'response' \
--save_every 20 \
--gpt_eval \
--api_key '' \
--model 'chatgpt-4o-latest' 
```

### Score Calculation

Finally, execute `python evaluation/calculate_acc.py` to calculate the final score based on the evaluation results. 
This step will compute overall accuracy as well as accuracy for each subject, category, and tasks.


## 📝Citation

If you find our benchmark useful in your research, please consider citing this BibTex:

```
@misc{hao2025mllmsreasonmultimodalityemma,
      title={Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark}, 
      author={Yunzhuo Hao and Jiawei Gu and Huichen Will Wang and Linjie Li and Zhengyuan Yang and Lijuan Wang and Yu Cheng},
      year={2025},
      eprint={2501.05444},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.05444}, 
}
```
