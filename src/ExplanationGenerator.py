#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inGen - Explainability module.

"""
import configparser

from langchain.llms import OpenAI
from langchain.chains import SinglePromptChain
from langchain_community.llms import Ollama

class ExplanationGenerator:
    def __init__(self, config_path=CONFIG_FILE_PATH):
        self.config = self.read_config(config_path)
        self.llm = self.initialize_llm()

    def read_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def initialize_llm(self):
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

    def generate_explanation(self, decision_context):
        prompt = self.create_prompt_from_context(decision_context)
        chain = SinglePromptChain(llm=self.llm, prompt=prompt)
        result = chain.run()
        return result.output

    def create_prompt_from_context(self, context):
        prompt = f"Explain the decision made in the following context: {context}"
        return prompt
