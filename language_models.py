import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as palm
import config
from concurrent.futures import ThreadPoolExecutor
import requests


# This file is modified based on the https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/language_models.py

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float):
        """
        Generates response by multi-threads for each requests
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self,
                         full_prompts_list,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        inputs = self.tokenizer(
            full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913,
            9092,
            16675])


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 120

    def __init__(self, model_name, api_key=config.OPENAI_API_KEY) -> None:
        self.model_name = model_name
        self.api_key = api_key
        openai.api_key = self.api_key
        if config.IS_USE_PROXY_OPENAI:
            openai.proxy = config.PROXY

        if config.IS_USE_CUSTOM_OPENAI_API_BASE:
            openai.api_base = config.OPENAI_API_BASE

    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                if "gpt" in self.model_name:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        request_timeout=self.API_TIMEOUT,
                    )
                    output = response["choices"][0]["message"]["content"]
                elif "text-davinci" in self.model_name:
                    # Convert conversation to prompt
                    response = openai.Completion.create(
                        engine=self.model_name,
                        prompt=conv,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        request_timeout=self.API_TIMEOUT,
                    )
                    output = response["choices"][0]["text"]

                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0,):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self.generate, convs_list, [
                                   max_n_tokens]*len(convs_list), [temperature]*len(convs_list), [top_p]*len(convs_list))
        return list(results)


class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 500

    def __init__(self, model_name, api_key=config.ANTHROPIC_API_KEY) -> None:
        self.model_name = model_name
        self.api_key = api_key
        if config.IS_USE_PROXY_OPENAI:
            self.model = anthropic.Anthropic(
                api_key=self.api_key,
                proxies=config.PROXY
            )
        else:
            self.model = anthropic.Anthropic(
                api_key=self.api_key,
            )

    def generate(self, conv: List,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_n_tokens,
                    prompt=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0,):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self.generate, convs_list, [
                                   max_n_tokens]*len(convs_list), [temperature]*len(convs_list), [top_p]*len(convs_list))
        return list(results)


class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last

                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[:(max_n_tokens*4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(1)
        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class OpenSourceModelAPI(LanguageModel):
    API = config.OPEN_SOURCE_MODEL_API

    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name == "vicuna-api":
            self.API = self.API + "/vicuna"
        elif self.model_name == "llama2-api":
            self.API = self.API + "/llama2"

    def batched_generate(self, conv: List,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        response = requests.post(self.API, json={
            "full_prompts_list": conv,
            "max_tokens": max_n_tokens,
            "temperature": temperature,
            "top_p": top_p
        })

        if response.status_code != 200:
            print('Request failed with status code:', response.status_code)

            return []

        return response.json()["output_list"]

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0,):

        return self.batched_generate(convs_list, max_n_tokens, temperature, top_p)
