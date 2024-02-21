#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inGen - Explainability module.

"""
import configparser
import requests
import openai

CONFIG_FILE_PATH = "config.ini"

class ExplanationClient:
    def __init__(self, api_key=None, api_url=None):
        self.api_key = api_key
        self.api_url = api_url

    def generate_explanation(self, prompt, max_tokens=100, temperature=0.7):
        raise NotImplementedError("This method should be implemented by subclasses.")

class OpenAIExplanationClient(ExplanationClient):
    def generate_explanation(self, prompt, max_tokens=100, temperature=0.7):
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].text.strip()

class OllamaExplanationClient(ExplanationClient):
    def generate_explanation(self, prompt, max_tokens=100, temperature=0.7):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        response = requests.post(self.api_url, json=payload, headers=headers)
        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("text", "").strip()

class ExplanationGenerator:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.client = self._initialize_client()

    def _initialize_client(self):
        # Dynamically select and initialize the appropriate API client based on config
        if 'OpenAI' in self.config:
            return OpenAIExplanationClient(api_key=self.config['OpenAI']['api_key'])
        elif 'Ollama' in self.config:
            return OllamaExplanationClient(api_url=self.config['Ollama']['api_url'],
                                           api_key=self.config['Ollama'].get('api_key'))
        else:
            raise ValueError("Unsupported API configuration.")

    def generate_explanation(self, prompt, max_tokens=100, temperature=0.7):
        return self.client.generate_explanation(prompt, max_tokens=max_tokens, temperature=temperature)

if __name__ == '__main__':
    # Initialize the generator with config file
    explanation_generator = ExplanationGenerator(config_file=config_file_path)

    # Generate explanations
    explanation = explanation_generator.generate_explanation("Explain why AI chose action A over B.", max_tokens=200)
    print(explanation)
