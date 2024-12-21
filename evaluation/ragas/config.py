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

import os
import sys
import logging
from pip._vendor import tomli
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

CONFIG_PATH = os.getenv('CONFIG_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), './config/config.toml'))

logger = logging.getLogger(__name__)


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error(f'Config file does not exist: {CONFIG_PATH}')
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        cfg = tomli.loads(f.read())
    
    return cfg


config = load_config()


def load_llm() -> BaseLanguageModel:
    models_config = config.get('models')
    llm_type = models_config.get('llm_type', 'openai')
    if llm_type == 'openai':
        os.environ["OPENAI_API_BASE"] = models_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = models_config.get('openai_api_key', '')

        from langchain_openai.chat_models import ChatOpenAI

        return ChatOpenAI(model=models_config.get('llm_model', 'gpt-3.5-turbo-16k'))
    
    elif llm_type == 'tongyi':
        os.environ["DASHSCOPE_API_KEY"] = models_config.get('dashscope_api_key', '')

        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(model=models_config.get('llm_model', 'qwen1.5-72b-chat'))

    elif llm_type == 'glm':
        os.environ["OPENAI_API_BASE"] = models_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = models_config.get('openai_api_key', '')
        from langchain_community.chat_models import ChatZhipuAI

        model = ChatZhipuAI(
            temperature=models_config.get('temperature', 1),
            api_key=models_config.get('openai_api_key', ''),
            model=models_config.get('llm_model', 'gpt-3.5-turbo-16k')
        )
        return model

    logger.error(f'Unsupported LLM model: {llm_type}')
    sys.exit(1)


def load_embeddings() -> Embeddings:
    embedding_config = config.get('embedding')
    emb_type = embedding_config.get('emb_type', 'openai')
    if emb_type == 'openai':
        os.environ["OPENAI_API_BASE"] = embedding_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = embedding_config.get('openai_api_key', '')

        from langchain_openai.embeddings import OpenAIEmbeddings
        
        return OpenAIEmbeddings(model=embedding_config.get('embeddings_model', 'text-embedding-ada-002'))
    
    elif emb_type == 'dashscope':
        os.environ["DASHSCOPE_API_KEY"] = embedding_config.get('dashscope_api_key', '')

        from langchain_community.embeddings.dashscope import DashScopeEmbeddings

        return DashScopeEmbeddings(model=embedding_config.get('embeddings_model', 'text-embedding-v2'))
    logger.error(f'Unsupported Embeddings model: {emb_type}')
    sys.exit(1)