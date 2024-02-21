#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inGen - Explainability module.

"""
import configparser

from langchain.llms import OpenAI
from langchain.chains import SinglePromptChain
from langchain_community.llms import Ollama

CONFIG_FILE_PATH = "config.ini"

class ExplanationGenerator:
    def __init__(self, llm='openai', api_key=None):
        self.llm = self.initialize_llm(llm, api_key)

    def initialize_llm(self, llm_name, api_key):
        if llm_name == 'openai':
            return OpenAI(api_key=api_key)
        elif llm_name == 'ollama':
            return Ollama(model="mistral")
        else:
            raise ValueError(f"Unsupported LLM: {llm_name}")

    def generate_explanation(self, decision_context):
        # Construct the prompt based on the decision context
        prompt = self.create_prompt_from_context(decision_context)

        # Use LangChain's SinglePromptChain for straightforward prompt-response interaction
        chain = SinglePromptChain(llm=self.llm, prompt=prompt)

        # Execute the chain to generate the explanation
        result = chain.run()
        return result.output

    def create_prompt_from_context(self, context):
        # This is a placeholder method. You should implement logic to
        # construct a meaningful prompt based on the AI decision context.
        # For example:
        prompt = f"Explain the decision made in the following context: {context}"
        return prompt

