# ChatTS: Time-Series Multimodal LLM
Code, datasets and model for `ChatTS`: [ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning](https://arxiv.org/pdf/2412.03104).

## Introduction
This repository provides several toolkits for generating synthetic data with the approaches introduced in `ChatTS`, as well as the evaluation code for reproduction:
- Toolkits for generating synthetic time series data and the corresponding attribues: `chatts/ts_generator.py`.
- Example code for generating a training dataset with pre-defined templates: `chatts/generate_template_qa.py`, which can be further used as seed QAs for TSEvol.
- Example code for generating a training dataset with LLMs: `chatts/generate_llm_qa`, which can be further used as seed QAs for TSEvol.
- Code implementation for `TSEvol` with the generated seed QAs: `chatts/evol/evol_instruct.py`.
- Code implementation for evaluation: `evaluation/`.
- A trained `ChatTS` model and evaluations datasets (Refer to the section below for more details).
- A simple demo for inference: `demo.ipynb`.


We also provide the evaluation datasets collected by us. You can download the evaluation datasets from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14349206.svg)](https://doi.org/10.5281/zenodo.14349206).
A fine-tuned `ChatTS` model have been open-sourced at [here](https://cloud.tsinghua.edu.cn/d/c23644020adc4d0fbc0a/). You may need to download the files one by one. You can download and try it!

## How To Use
### Installation
- Basic requirements for model inference: `python>=3.11`, `deepspeed`, `vllm`, `flash-attn` (refer to `requirements.txt`).
- Download the evaluation datasets from [here](https://doi.org/10.5281/zenodo.14349206) and put them under `evaluation/dataset` (`evaluation/dataset/dataset_a.json` and `evaluation/dataset/dataset_b.json`).
- Download the trained model weights from [here](https://cloud.tsinghua.edu.cn/d/c23644020adc4d0fbc0a/), extract it and put all the extracted files under `ckpt/` (`ckpt/config.json`, etc).
- **Note:** `ChatTS` is trained based on a 14B-sized base model, so you need to ensure that you have a GPU with sufficient memory for inference. Additionally, due to the model's requirements, `Flash-Attention` (https://github.com/Dao-AILab/flash-attention) is essential, so you need to ensure that your GPU meets the installation requirements for Flash-Attention. Recommended GPUs: A100/A800.

### Training Data Generation
1. **QA Generation with Templates**. Use `python3 -m chatts.generate_template_qa` to generate a training dataset with pre-defined templates.
2. **QA Generation with LLMs**. You need a downloaded LLM that can be loaded with `vLLM` to perform this step. Set `[LOCAL_LLM_PATH]` in `chatts/generate_llm_qa.py` to a local LLM model (e.g., QWen2.5-32B-Instruct, **NOT ChatTS Model**) and set num_gpus, gpu_per_model accordingly. Use `python3 -m chatts.generate_llm_qa` to generate a training dataset with LLMs.
3. **TSEvol**. You need a downloaded LLM that can be loaded with `vLLM` to perform this step. The datasets generated in Step 1 and Step 2 will be used as seed QAs in TSEvol, so please make sure that you have successfully generated the previous datasets before running TSEvol. Then, refer to the steps in `chatts/evol/evol_instruct.py`:
    1. Set `[LOCAL_LLM_PATH]` in `evol_instruct.py` to the path of a local LLM model (e.g., QWen2.5-32B-Instruct. **NOT ChatTS Model**) for QA generation and set num_gpus, gpu_per_model accordingly in `chatts/evol/evol_instruct.py`.
    2. Run `python3 -m chatts.evol.evol_instruct`.
    3. The output will be saved to the file specified in `OUTPUT_FILE`.

### Try the ChatTS Model
- Following the steps in `Installation` to download the trained `ChatTS` model and place it under `ckpt`. 
- The ChatTS model can be loaded directly using the `transformers` library. However, due to the time series data as input, the API usage differs from the standard implementation. **Refer to `demo.ipynb` for more information.**

### Deepspeed Model Inference for Evaluation
- We provide a simple script for inference of ChatTS (`chatts/inference_tsmllm_deepspeed.py`) with `deepspeed`. After installing `deepspeed`, please set the `WORKDIR` (the absolute path of the current directory) and the evaluation dataset in the script. Then, run the following command to do the model inference:
```sh
deepspeed --num_gpus [YOUR_NUM_GPUS] --master_port 12345 chatts/inference_tsmllm_deepspeed.py
```

### Evaluation
- Install `ragas==0.1.9` (https://github.com/explodinggradients/ragas), which is used for evaluating the inductive reasoning results.
- Set the `API_KEY` and `OPENAI_URL` in `evaluation/ragas/config/config.toml` (Refer to https://platform.openai.com/docs/api-reference).
- Run `python3 -m evaluation.evaluate_tsmllm_models` to evaluate `ChatTS` (make sure you have done the model inference before).
- We also provide a simple demo to evaluate the performance of text-based GPT models. After setting your `API_KEY` and `OPENAI_URL` in `evaluation/evaluate_gpt_text_models.py`, run the command `python3 -m evaluation.evaluate_gpt_text_models` to obtain the evaluation results of the text-based GPT model.

## Evaluation Datasets
- We provide the two evaluation datasets we collected, as mentioned in the paper, located in the `evaluation/dataset` folder. Each dataset sample contains the following fields: `timeseries`, `question`, `answer` (standard answers in text format, for reference only), `attributes` (structured labels used for result evaluation), and `ability_types` (the tasks included in the question). **Please note that, to reduce the evaluation cost, we have merged different questions for the same time series into a single `question`, using numbering to distinguish between the different questions.** Therefore, the actual number of questions in the evaluation dataset may be greater than the number of `timeseries` entries. Additionally, please note that some tasks in inductive reasoning and alignment are grouped into the same question, as the inductive reasoning tasks involve explaining the physical meanings of time series attributes.
- The `MCQ2` dataset is a third-party open-source dataset, we do not provide it in this repository. Please download it directly via https://github.com/behavioral-data/TSandLanguage.

## Notes
- You can use the CPU for inference. However, since our current ChatTS model does not implement `kv_cache` (which we plan to implement shortly), the inference speed may be significantly slow.
- The code, data, and models in this repository are for review purposes only. Due to company policy, the open-source code is currently under review. Once the review is approved, we will release it on GitHub and the HuggingFace platform for public access.

## Reference
- QWen (https://github.com/QwenLM/Qwen2.5)
- DeepSpeed (https://www.deepspeed.ai/)
- RAGAS (https://github.com/explodinggradients/ragas)
- VLLM (https://github.com/vllm-project/vllm)
- Flash Attention (https://github.com/Dao-AILab/flash-attention)

## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue for a security vulnerability.

## License
This project is licensed under the [MIT License](LICENSE).
