# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
from loguru import logger
import numpy as np
import json
import os
from tqdm import tqdm
import traceback
from typing import *

from evaluation.evaluate_qa import evaluate_batch_qa
from multiprocessing import Pool

# CONFIG
MODEL = 'gpt-4o-mini'
EXP = 'gpt-4o-mini-dataset-a'
DATASET = 'evaluation/dataset/dataset_a.json'
OPENAI_API_KEY = "[Your API Key]"
OPENAI_BASE_URL = "[Your Base URL]"
NUM_WORKERS = 32


def ask_gpt_api_with_timeseries(case_idx: int, timeseries: np.ndarray, cols: List[str], question: str) -> str:
    openai.api_key = OPENAI_API_KEY
    openai.base_url = OPENAI_BASE_URL

    client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    prompt_list = question.split('<ts><ts/>')
    prompt = prompt_list[0]
    for ts in range(len(timeseries)):
        cur_ts = ','.join([f"{i:.2f}" for i in timeseries[ts]])
        prompt += f"{cur_ts}" + prompt_list[ts + 1]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]

    timeout_cnt = 0
    while True:
        if timeout_cnt > 10:
            logger.error("Too many timeout!")
            raise RuntimeError("Too many timeout!")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                timeout=20
            )
            break
        except Exception as err:
            logger.error(err)
            logger.error("API timeout, trying again...")
            timeout_cnt += 1

    answer = response.choices[0].message.content
    total_tokens = response.usage.prompt_tokens
    return answer, total_tokens

def process_sample(args):
    sample, idx = args
    try:
        timeseries = sample['timeseries']
        cols = sample['cols']
        question_text = sample['question']
        label = sample['answer']

        answer, total_tokens = ask_gpt_api_with_timeseries(idx, timeseries, cols, question_text)

        return {
            'idx': idx,
            'question_text': question_text,
            'response': answer,
            'num_tokens': total_tokens
        }
    except Exception as err:
        logger.error(err)
        traceback.print_exc()
        return None


if __name__ == '__main__':
    dataset = json.load(open(DATASET))

    generated_answer = []
    os.makedirs(f'exp/{EXP}', exist_ok=True)
    if os.path.exists(f"exp/{EXP}/generated_answer.json"):
        generated_answer = json.load(open(f"exp/{EXP}/generated_answer.json"))
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info("Start Generation...")
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), total=len(idx_to_generate)))

    # Filter out None results and update generated_answer
    generated_answer.extend([res for res in results if res is not None])
    json.dump(generated_answer, open(f"exp/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)

    # Evaluation
    evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=NUM_WORKERS)
