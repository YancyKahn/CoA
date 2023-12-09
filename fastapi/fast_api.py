import uvicorn
import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import Dict, List
import gc

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


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
        self.tokenizer

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        try:
            inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
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
                    temperature=1, # To prevent warning messages
                )
                
            # If the model is not an encoder-decoder type, slice off the input tokens
            if not self.model.config.is_encoder_decoder:
                output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

            # Batch decoding
            outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        except RuntimeError as e:
            print(e)
            try:
                del inputs
            except:
                pass

            try:
                del output_ids
            except: 
                pass

            gc.collect()
            torch.cuda.empty_cache()

            return None

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
            # self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])


def load_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    use_fast=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                        device_map="auto")
    
    if 'llama-2' in model_name.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'vicuna' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HuggingFace(model_name, model, tokenizer)

    return lm

def main(args):

    app = FastAPI()
    host = args.host
    port = args.port
    log_level = args.log_level
    lms = {}
    for model_name in args.model_name:
        if "llama-2" in model_name.lower():
            lms["llama2"] = load_tokenizer_and_model(model_name)
        if "vicuna" in model_name.lower():
            lms["vicuna"] = load_tokenizer_and_model(model_name)

    @app.post("/generate/llama2")
    async def generate(request: Request):
        # Get request data
        data = await request.json()

        full_prompts_list = data.get("full_prompts_list", [])
        max_n_tokens = data.get("max_n_tokens", 1024)
        temperature = data.get("temperature", 0.9)
        top_p = data.get("top_p", 1.0)


        output_list = lms["llama2"].batched_generate(full_prompts_list, max_n_tokens, temperature, top_p)

        return {"output_list": output_list}

    @app.post("/generate/vicuna")
    async def generate(request: Request):
        # Get request data
        data = await request.json()

        full_prompts_list = data.get("full_prompts_list", [])
        max_n_tokens = data.get("max_n_tokens", 1024)
        temperature = data.get("temperature", 0.9)
        top_p = data.get("top_p", 1.0)


        output_list = lms["vicuna"].batched_generate(full_prompts_list, max_n_tokens, temperature, top_p)

        return {"output_list": output_list}

    
    uvicorn.run(app, host=host, port=port, log_level=log_level)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##### Model Settings #####
    parser.add_argument("--model-name", 
                        type=str,
                        nargs="+",
                        default=["lmsys/vicuna-13b-v1.5-16k"],
                        choices=["lmsys/vicuna-7b-v1.5", 
                                 "lmsys/vicuna-13b-v1.5-16k",
                                 "meta-llama/Llama-2-7b-chat-hf",
                                 "meta-llama/Llama-2-13b-chat-hf",
                                 "lmsys/vicuna-13b-v1.5"],
                        help="model_name")

    #### FastAPI Settings ####
    parser.add_argument("--host", 
                        type=str, 
                        default="0.0.0.0", 
                        help="host")
    
    parser.add_argument("--port",
                        type=int,
                        default=9999,
                        help="port")
    
    parser.add_argument("--log_level",
                        type=str,
                        default="info",
                        help="log_level")
    
    args = parser.parse_args()

    print(args)
    
    main(args)
