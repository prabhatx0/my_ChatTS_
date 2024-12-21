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
# Reference (RAGAS): https://github.com/explodinggradients/ragas

import logging
import difflib
import typing as t
from dataclasses import dataclass, field
import multiprocessing
import numpy as np
from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM
from filelock import FileLock
from ragas.run_config import RunConfig
from evaluation.ragas.config import config
import json
import os

logger = multiprocessing.get_logger()

# Path to cache file
DATA_DIR = config.get('data_dir', 'data').rstrip('/')
CACHE_PATH = os.path.join(DATA_DIR, 'gt_cache.json')
CACHE_LOCK_PATH = CACHE_PATH.replace('.json', '.lock')

def load_cache():
    # logger.info("Loading cache")
    if os.path.exists(CACHE_PATH):
        with FileLock(CACHE_LOCK_PATH):
            with open(CACHE_PATH, 'r') as f:
                return json.load(f)
    return {}

def save_cache(cache):
    # logger.info("Saving cache")
    with FileLock(CACHE_LOCK_PATH):
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r') as f:
                cache.update(json.load(f))
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

async def get_gt_keywords(llm, question, groundtruth, callbacks, is_async):
    cache = load_cache()
    cache_key = json.dumps(question, ensure_ascii=False) + '|||' + json.dumps(groundtruth, ensure_ascii=False)

    if cache_key in cache:
        return cache[cache_key]

    # Use the LLM to generate gt_keywords
    gt_keywords = await generate_gt_keywords_with_llm(llm, question, groundtruth, callbacks, is_async)

    cache[cache_key] = gt_keywords
    save_cache(cache)

    return gt_keywords

async def generate_gt_keywords_with_llm(llm, question, groundtruth, callbacks, is_async):
    prompt = GT_PROMPT.format(**{
        "question": question,
        "ground_truth": groundtruth
    })
    response = llm.generate([prompt], callbacks=callbacks)
    response = await json_loader.safe_load(
        response.generations[0][0].text, llm, is_async=is_async
    )
    if 'gt_keywords' in response and type(response['gt_keywords']) == list:
        result = []
        # Remove keywords from questions
        for key in response['gt_keywords']:
            if key.lower() not in question.lower() or key.lower() in groundtruth.lower():
                result.append(key)
        return result
    else:
        return []

GT_PROMPT = Prompt(
    name="gt_keywords",
    instruction="""Given a question and the ground truth, extract the following information:
            "gt_keywords": Identify and return a list of keywords or phrases contained in the ground_truth. This list should only include the minimal combination of key points that directly answer the question, avoiding words unrelated to the question as much as possible. Only **1 or 2** keywords are needed in each list, and the keywords should be as concise as possible and **as short as possible**.
            """,
    examples=[
        {
            "question": """What protocol does AMF use to ensure correct time?""",
            "ground_truth": """The role of NTP is to synchronize the time of all clocked devices within the network, ensuring that the clock times of all devices in the network remain basically consistent, so that the devices can provide various applications based on unified time. Possible applicable network elements: AMF, MME, SGSN""",
            "Extracted statements": {
                "gt_keywords": ["NTP"]
            },
        },
        {
            "question": """What is the purpose of MME assigning a TA List to UE?""",
            "ground_truth": """The purpose of MME assigning a TA List to UE is to manage the location of idle UEs. After MME assigns a TA List to UE, the UE will not initiate a TA update to MME when moving within this TA List. When the UE moves out of the TA List range, it will initiate a TA update, and the MME will know that the location of the idle UE is within the TA List range. When paging is needed, it can be done within the TA List range.""",
            "Extracted statements": {
                "gt_keywords": ["manage the location of idle UEs"]
            },
        }
    ],
    input_keys=["question", "ground_truth"],
    output_key="Extracted statements",
    output_type="json",
)


ANSWER_PROMPT = Prompt(
    name="answer_correctness_step_2",
    instruction="""Given a question, an answer generated by the model, and the list of ground truth keywords (gt_keywords), extract the following information:
        "overlapping_keywords": From the list of "gt_keywords", identify any terms or phrases that also appear in the model's answer. These overlapping keywords indicate the points of agreement or coverage between the model's answer and the ground truth.""",
    examples=[
        {
            "question": "What powers the sun and what is its primary function?",
            "gt_keywords": ["nuclear fusion", "energy", "light", "essential for life", "climate system", "weather", "ocean currents"],
            "answer": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.",
            "Extracted statements": {
                "overlapping_keywords": ["light"]
            }
        },
        {
            "question": "What is the boiling point of water?",
            "gt_keywords": ["100 degrees Celsius", "212 degrees Fahrenheit", "sea level", "change with altitude"],
            "answer": "The boiling point of water is 100 degrees Celsius at sea level.",
            "Extracted statements": {
                "overlapping_keywords": ["100 degrees Celsius", "sea level"]
            }
        },
        {
            "question": "What information should be submitted when contacting technical support for a communication technology company?",
            "gt_keywords": ["fault details", "log files and alarm query results", "steps taken to address the issue", "commands executed", "results of those actions", "remote access details", "contact information for relevant personnel"],
            "answer": "When contacting technical support for a communication technology company, the following information should be provided: 1. Fault details: Time, location, and event description.",
            "Extracted statements": {
                "overlapping_keywords": ["fault details"]
            }
        },
        {
            "question": "What are the benefits of a balanced diet?",
            "gt_keywords": ["provides essential nutrients", "maintains a healthy weight", "reduces risk of chronic diseases", "supports overall health"],
            "answer": "A balanced diet helps maintain a healthy weight and supports overall health.",
            "Extracted statements": {
                "overlapping_keywords": ["maintain a healthy weight", "supports overall health"]
            }
        },
        {
            "question": "What is the capital of France?",
            "gt_keywords": ["Paris"],
            "answer": "The capital of France is Paris.",
            "Extracted statements": {
                "overlapping_keywords": ["Paris"]
            }
        },
        {
            "question": "How does photosynthesis work?",
            "gt_keywords": ["process by which plants convert sunlight into energy", "involves chlorophyll", "produces oxygen", "occurs in the chloroplasts"],
            "answer": "Photosynthesis is the process by which plants use sunlight to produce energy and oxygen.",
            "Extracted statements": {
                "overlapping_keywords": ["process by which plants convert sunlight into energy", "produces oxygen"]
            }
        },
    ],
    input_keys=["question", "gt_keywords", "answer"],
    output_key="Extracted statements",
    output_type="json",
)


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings):
    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    answer_prompt: Prompt = field(default_factory=lambda: ANSWER_PROMPT)
    weights: list[float] = field(default_factory=lambda: [1.0, 0.0])
    keyword_matching_threshold: float = field(default_factory=lambda: 0.6)
    answer_similarity: AnswerSimilarity | None = None

    def __post_init__(self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )
        self.answer_detail = {}

    def _compute_statement_presence(self, gt_keywords, prediction: t.Any, question: str) -> float:
        assert self.llm is not None, "LLM must be set"

        key_map = [
            "overlapping_keywords",
        ]
        if prediction:
            try:
                prediction = [prediction.get(k, np.nan) for k in key_map]
                overlapping_keywords = [
                    item if isinstance(item, list) else np.nan for item in prediction
                ][0]
                if gt_keywords is None or (type(gt_keywords) == float and np.isnan(gt_keywords)):
                    logger.warning('[gt_keywords] gt_keywords is nan!')
                    gt_keywords = []
                gt_keywords = [k.lower() for k in gt_keywords]
                overlapping_keywords = [k.lower() for k in overlapping_keywords]
                self.answer_detail[question] = {'answer_keywords': '|'.join(overlapping_keywords)}
                overlapping_keywords = [k for k in overlapping_keywords if self.match(gt_keywords, k)]
                num_ok = len(overlapping_keywords)
                num_all = len(gt_keywords)
                self.answer_detail[question].update({
                    'gt_keywords': '|'.join(gt_keywords),
                    'overlapping_keywords': '|'.join(overlapping_keywords),
                    'num_ok': int(num_ok),
                    'num_all': int(num_all)
                })
                if any([np.isnan(i) for i in [num_ok, num_all]]):
                    score = np.nan
                    logger.warning(
                        "Invalid prediction format. Expected a list of dictionaries with keys 'gt_keywords', 'overlapping_keywords'"
                    )
                else:
                    score = min(num_ok / num_all, 1.0) if num_all > 0 else np.nan
            except:
                score = np.nan
        else:
            score = np.nan

        return score

    def match(self, arr, k):
        for item in arr:
            if difflib.SequenceMatcher(None, item, k).ratio() >= self.keyword_matching_threshold:
                return True
        return False

    async def _ascore(self, row: dict, callbacks: t.Any, is_async: bool) -> float:
        assert self.llm is not None, "LLM must be set"

        q, a, g = row["question"], row["answer"], row["ground_truth"]

        if len(row["answer"].strip('"')) == 0:
            return 0.0

        gt_keywords = await get_gt_keywords(self.llm, q, g, callbacks, is_async=is_async)
 
        prompt_input = {
            "question": q,
            "gt_keywords": gt_keywords,
            "answer": a
        }
        
        p_value = self.answer_prompt.format(**prompt_input)

        is_statement_present = self.llm.generate(
            [p_value], callbacks=callbacks
        )

        prediction = await json_loader.safe_load(
            is_statement_present.generations[0][0].text, self.llm, is_async=is_async
        )
        f1_score = self._compute_statement_presence(gt_keywords, prediction, q)

        if self.weights[1] == 0:
            similarity_score = 0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            callbacks = []
            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks, is_async=is_async
            )
        if q not in self.answer_detail:
            self.answer_detail[q] = {}
        self.answer_detail[q]['answer_similarity'] = float(similarity_score)
        self.answer_detail[q]['keywords_prediction'] = str(is_statement_present.generations[0][0].text)
        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "llm must be set to compute score"

        logger.info(f"Adapting AnswerCorrectness metric to {language}")
        self.correctness_prompt = self.answer_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.answer_prompt.save(cache_dir)


answer_correctness = AnswerCorrectness()
