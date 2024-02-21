#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inGen - Explainability module.

This module provides the functionality to generate explanations for system
adaptations based on log entries. It is intended to be used as a part of the
inGen system, which is a conversational AI system that can adapt to user
feedback.

Example of configuration file (config.ini):

    [General]
    llm = ollama
    use_case_context = A company is using an edge-cloud computing infrastructure to process data from IoT devices spread across multiple locations. The primary intent is to optimize energy consumption across the infrastructure while ensuring data is processed efficiently and sustainably.
    system_prompt =	You are a helpful assistant that describes and explains adaptations made in the edge-cloud computing infrastructure based on the available information.

    [OpenAI]
    api_key = your_openai_api_key_here

    [Ollama]
    model = mistral

Example of a log file:

    2024-02-20T09:00:00.000Z [inChat] INFO: User session initiated.
    2024-02-20T09:01:15.432Z [inChat] INFO: Intent received - Intent ID: 001 - "Aim to reduce energy consumption by 20% in edge devices, ensuring data processing latency is not compromised."
    2024-02-20T09:01:20.876Z [IntentProcessor] INFO: Intent ID: 001 processed - "Reduce energy consumption by 20%" extracted and passed to EnergyOptimizationAgent.
    2024-02-20T09:01:25.123Z [EnergyOptimizationAgent] INFO: Analyzing current energy consumption patterns for Intent ID: 001.
    2024-02-20T09:02:30.456Z [EnergyOptimizationAgent] INFO: Workload distribution adjustment initiated for Intent ID: 001 to prioritize low-energy-consuming tasks during peak hours.
    2024-02-20T09:03:45.789Z [AlgorithmOptimizationAgent] INFO: Implementing energy-efficient data processing algorithm on edge devices (ID: EdgeNode-03, EdgeNode-07) for Intent ID: 001.

    2024-02-20T09:05:00.123Z [inChat] INFO: Intent received - Intent ID: 002 - "Minimize carbon footprint of cloud-based processing by aligning with renewable energy peak production hours."
    2024-02-20T09:05:05.678Z [IntentProcessor] INFO: Intent ID: 002 processed - "Minimize carbon footprint" extracted and passed to CloudSchedulingAgent.
    2024-02-20T09:05:10.891Z [CloudSchedulingAgent] INFO: Adjusting task scheduling to utilize renewable energy peaks for Intent ID: 002 based on predictions.
    2024-02-20T09:06:00.234Z [CloudSchedulingAgent] INFO: Non-urgent cloud-based data analysis tasks rescheduled (ID: CloudTask-112, CloudTask-118) for Intent ID: 002.

    2024-02-20T09:07:15.432Z [EnergyOptimizationAgent] INFO: Adaptation for Intent ID: 001 - Energy consumption on edge devices reduced by 15% without significant impact on latency.
    2024-02-20T09:07:20.876Z [SustainabilityMetricsAgent] INFO: Adaptation for Intent ID: 002 - Carbon footprint of cloud processing reduced by aligning 60% of tasks with renewable energy availability.

    2024-02-20T09:08:30.456Z [MonitoringAgent] INFO: Continuous monitoring initiated for further optimization opportunities related to Intent ID: 001 and Intent ID: 002.
    2024-02-20T09:10:45.789Z [FeedbackAgent] INFO: Preparing explanation report for stakeholders based on adaptations made for Intent ID: 001 and Intent ID: 002.
    2024-02-20T09:12:00.123Z [inChat] INFO: Explanation report sent to stakeholders detailing adaptations made and progress towards intents for Intent ID: 001 and Intent ID: 002.

"""
import configparser
import re

from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

CONFIG_FILE_PATH = "config/config.ini"
DATA_FILE_PATH = "data/"
LOG_FILE_PATH = "data/system_adaptions.log"

class ExplanationGenerator:
    """Generates explanations for system adaptations based on log entries.

    This class provides the functionality to generate explanations for system
    adaptations based on log entries. It is intended to be used as a part of the
    inGen system, which is a conversational AI system that can adapt to user
    feedback.

    Attributes:
        config (configparser.ConfigParser): The configuration settings for the
            explanation generator.
        log_file_path (str): The path to the log file containing system
            adaptations.
        system_prompt (str): The prompt to use for the language model.
        llm (langchain.llms.LanguageModel): The language model to use for
            generating explanations.

    """

    def __init__(self, config_path=CONFIG_FILE_PATH, log_file_path=LOG_FILE_PATH):
        """Initializes the ExplanationGenerator.

        Args:
            config_path (str): The path to the configuration file for the
                explanation generator.
            log_file_path (str): The path to the log file containing system
                adaptations.

        """

        self.config = self._read_config(config_path)
        self.log_file_path = log_file_path
        self.llm = self._initialize_llm()


    def _read_config(self, config_path):
        """
        Reads the configuration settings from the specified file.

        Args:
            config_path (str): The path to the configuration file.

        """
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _initialize_llm(self):
        """Initializes the language model to use for generating explanations.

        Returns:
            langchain.llms.LanguageModel: The language model to use for
                generating explanations.

        """
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

    def generate_explanation(self, intent_id):
        """Generates an explanation for system adaptations based on log entries.

        This method uses an LLM to generate an explanation for system adaptations
        based on log entries. It reads the log entries from the log file and
        extracts the intent and adaptation information. It then generates a
        prompt for the LLM to generate an explanation and uses the LLM to
        generate the explanation.

        Args:
            intent_id (str): The intent ID for which to generate an explanation.

        Returns:
            str: The generated explanation.

        """
        log_entries = self._read_log_entries()
        intent_entries = self._extract_intent_entries(log_entries)
        intent_adaptations = self._extract_intent_adaptations(log_entries)
        intent_text = intent_adaptations[intent_id]['intent']
        adaptations = intent_adaptations[intent_id]['adaptations']

        prompt = self._generate_explanation_prompt(log_entries)
        chain = prompt | self.llm
        result = chain.invoke({})
        return result

    def _read_log_entries(self):
        """Reads the log entries from the log file.

        Returns:
            list: The log entries.

        """
        with open(self.log_file_path, "r") as file:
            log_entries = file.readlines()
        return log_entries

    def _generate_explanation_prompt(self, log_entries):
        """Generates a prompt for the language model to generate an explanation.

        Returns:
            str: The generated prompt.

        """
        use_case_context = self.config.get('General', 'use_case_context', fallback='')
        system_prompt = self.config.get('General', 'system_prompt', fallback='')
        full_system_prompt = f"{use_case_context}\n\n{system_prompt}"

        messages = [("system", full_system_prompt)]
        messages += [("user", entry) for entry in log_entries]

        prompt = ChatPromptTemplate.from_messages(messages)

        return prompt

    def _extract_intent_entries(self, log_entries):
        """Extracts the intent entries from the log entries.

        Args:
            log_entries (list): The log entries.

        Returns:
            dict: The extracted intent entries.

        """
        intent_entries = {}
        for entry in log_entries:
            match = re.search(r"Intent ID: (\d+) - \"(.+)\"", entry)
            if match:
                intent_id = match.group(1)
                intent_text = match.group(2)
                intent_entries[intent_id] = intent_text
        return intent_entries

    def _extract_intent_adaptations(self, log_entries):
        """Extracts the intent adaptations from the log entries.

        Args:
            log_entries (list): The log entries.

        Returns:
            dict: The extracted intent adaptations.

        """
        intent_adaptations = {}
        for entry in log_entries:
            match = re.search(r"Adaptation for Intent ID: (\d+) - (.+)", entry)
            if match:
                intent_id = match.group(1)
                adaptation = match.group(2)
                if intent_id not in intent_adaptations:
                    intent_adaptations[intent_id] = {"intent": None, "adaptations": []}
                if "Intent ID: " in adaptation:
                    intent_adaptations[intent_id]["intent"] = adaptation
                else:
                    intent_adaptations[intent_id]["adaptations"].append(adaptation)
        return intent_adaptations

    def generate_html_explanation(self):
        """Generates an HTML explanation for system adaptations based on log entries.

        This method uses an LLM to generate an explanation for system adaptations
        based on log entries and formats the explanation as an HTML document.

        Returns:
            str: The generated HTML explanation.

        """
        explanation = self.generate_explanation("001")
        html_explanation = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>System Adaptations Explanation</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                }}
                h1 {{
                    font-size: 24px;
                    margin-bottom: 16px;
                }}
                p {{
                    font-size: 16px;
                    margin-bottom: 8px;
                }}
            </style>
        </head>
        <body>
            <h1>System Adaptations Explanation</h1>
            <p>{explanation}</p>
        </body>
        </html>
        """
        return html_explanation



if __name__ == '__main__':
    eg = ExplanationGenerator()

    explanation = eg.generate_explanation("001")

    html_explanation = eg.generate_html_explanation()

    with open('explanation.html', 'w') as file:
        file.write(html_explanation)

