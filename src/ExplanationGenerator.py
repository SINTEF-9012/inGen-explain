#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inGen - Explainability module.

"""
import configparser

from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

CONFIG_FILE_PATH = "src/config.ini"

class ExplanationGenerator:
    def __init__(self, config_path=CONFIG_FILE_PATH):
        self.config = self.read_config(config_path)
        self.system_prompt = "You are a helpful AI assistant."
        self.llm = self._initialize_llm()

    def read_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _initialize_llm(self):
        llm_name = self.config.get('General', 'llm', fallback='openai')
        if llm_name == 'openai':
            api_key = self.config.get('OpenAI', 'api_key', fallback=None)
            if not api_key:
                raise ValueError("API key for OpenAI is required in config.ini")
            return OpenAI(api_key=api_key)
        elif llm_name == 'ollama':
            model = self.config.get('Ollama', 'model', fallback='mistral')
            return Ollama(model=model)
        else:
            raise ValueError(f"Unsupported LLM: {llm_name}")

    def generate_explanation(self, user_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{user_prompt}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({"user_prompt": user_prompt})
        return result

if __name__ == '__main__':

    eg = ExplanationGenerator()
    eg.generate_explanation("Why is the sky blue?")
