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
        self.system_prompt = "Please explain the system adaptations made based on the log entries."
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

    def generate_explanation(self):
        """Generates an explanation for system adaptations based on log entries.

        Returns:
            str: The generated explanation.

        """
        log_entries = self._read_log_entries()
        explanation = self._generate_explanation(log_entries)
        return explanation

    def _read_log_entries(self):
        """Reads the log entries from the log file.

        Returns:
            list: The log entries.

        """
        with open(self.log_file_path, "r") as file:
            log_entries = file.readlines()
        return log_entries

    def _generate_explanation(self, log_entries):
        """Generates an explanation for system adaptations based on log entries.

        Args:
            log_entries (list): The log entries.

        Returns:
            str: The generated explanation.

        """
        intent_entries = self._extract_intent_entries(log_entries)
        explanation = ""
        for entry in intent_entries:
            intent_id = self._extract_intent_id(entry)
            adaptation = self._extract_adaptation(entry)
            explanation += f"System adaptation for Intent ID {intent_id}: {adaptation}\n"

        return explanation

    def _extract_intent_id(self, entry):
        """Extracts the intent ID from a log entry.

        Args:
            entry (str): The log entry.

        Returns:
            str: The intent ID.

        """
        match = re.search(r"Intent ID: (\d+)", entry)
        if match:
            return match.group(1)
        return ""


    def _extract_intent_entries(self, log_entries):
        """Extracts the log entries related to system adaptations.

        Args:
            log_entries (list): The log entries.

        Returns:
            list: The log entries related to system adaptations.

        """
        intent_entries = []
        for entry in log_entries:
            if "Adaptation for Intent ID" in entry:
                intent_entries.append(entry)
        return intent_entries

    def _extract_adaptation(self, entry):
        """Extracts the adaptation from a log entry.

        Args:
            entry (str): The log entry.

        Returns:
            str: The adaptation.

        """
        match = re.search(r"Adaptation for Intent ID: \d+ - (.+)", entry)
        if match:
            return match.group(1)
        return ""

    def generate_explanation_prompt(self):
        """Generates a prompt for the language model to generate an explanation.

        Returns:
            str: The generated prompt.

        """
        # Construct the conversation prompt with system and user messages
        messages = [("system", self.system_prompt)]
        messages += [("user", entry) for entry in log_entries]

        prompt = ChatPromptTemplate.from_messages(messages)
        prompt = ChatPromptTemplate(self.system_prompt)
        return prompt

    def _extract_intent_adaptations(self, log_entries):
        """Extracts and organizes intent and adaptation information.

        Args:
            log_entries (list): The log entries.

        Returns:
            dict: A dictionary where each key is an intent ID and each value is a
                  dictionary with 'intent' and 'adaptations' keys.
        """
        intent_adaptations = {}
        for entry in log_entries:
            if "Intent received" in entry:
                intent_id = self._extract_intent_id(entry)
                intent_text = self._extract_intent_text(entry)
                intent_adaptations[intent_id] = {'intent': intent_text, 'adaptations': []}
            elif "Adaptation for Intent ID" in entry:
                intent_id = self._extract_intent_id(entry)
                adaptation_text = self._extract_adaptation(entry)
                if intent_id in intent_adaptations:
                    intent_adaptations[intent_id]['adaptations'].append(adaptation_text)
        return intent_adaptations

    def _extract_intent_text(self, entry):
        """Extracts the intent text from a log entry.

        Args:
            entry (str): The log entry.

        Returns:
            str: The intent text.
        """
        match = re.search(r'Intent ID: \d+ - "(.+)"', entry)
        if match:
            return match.group(1)
        return ""


    def generate_html_explanation(self):
        """Generates an HTML document summarizing intents and adaptations.

        The HTML document has CSS styles to make the document look modern and
        readable.

        Returns:
            str: HTML content.
        """
        log_entries = self._read_log_entries()
        intent_adaptations = self._extract_intent_adaptations(log_entries)

        # html_content = '<html><head><title>System Adaptations Explanation</title></head><body>'
        # html_content += '<h1>Explanation of System Adaptations</h1>'

        # for intent_id, details in intent_adaptations.items():
        #     html_content += f'<h2>Intent ID: {intent_id}</h2>'
        #     html_content += f'<p><strong>Intent:</strong> {details["intent"]}</p>'
        #     html_content += '<p><strong>Adaptations:</strong></p><ul>'
        #     for adaptation in details['adaptations']:
        #         html_content += f'<li>{adaptation}</li>'
        #     html_content += '</ul>'

        # html_content += '</body></html>'

        html_content = f"""
        <html>
        <head>
            <title>System Adaptations Explanation</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: auto;
                    max-width: 800px;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                h2 {{
                    color: #333;
                }}
                p {{
                    color: #333;
                }}
                ul {{
                    list-style-type: none;
                    padding: 0;
                }}
                li {{
                    margin-bottom: 10px;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
            </style>
        </head>
        <body>
            <h1>Explanation of System Adaptations</h1>
        """
        
        for intent_id, details in intent_adaptations.items():
            html_content += f"""
            <h2>Intent ID: {intent_id}</h2>
            <p><strong>Intent:</strong> {details["intent"]}</p>
            <p><strong>Adaptations:</strong></p>
            <ul>
            """
            for adaptation in details['adaptations']:
                html_content += f'<li>{adaptation}</li>'
            html_content += '</ul>'

        html_content += '</body></html>'

        return html_content

if __name__ == '__main__':
    eg = ExplanationGenerator()

    explanation = eg.generate_explanation()

    html_explanation = eg.generate_html_explanation()

    with open('explanation.html', 'w') as file:
        file.write(html_explanation)

