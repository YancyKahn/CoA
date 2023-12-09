
import common
from language_models import GPT, Claude, PaLM, HuggingFace, OpenSourceModelAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P

# This file is modified based on the https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/conversers.py

def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name=args.attack_model,
                        max_n_tokens=args.attack_max_n_tokens,
                        max_n_attack_attempts=args.max_n_attack_attempts,
                        temperature=ATTACK_TEMP,  # init to 1
                        top_p=ATTACK_TOP_P,  # init to 0.9
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name=args.target_model,
                        max_n_tokens=args.target_max_n_tokens,
                        temperature=TARGET_TEMP,  # init to 0
                        top_p=TARGET_TOP_P,  # init to 1
                        preloaded_model=preloaded_model,
                        )
    return attackLM, targetLM


class AttackLM():
    """
        Base class for attacker language models.

        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 max_n_attack_attempts: int,
                 temperature: float,
                 top_p: float):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

        if ("vicuna" in model_name or "llama" in model_name) and "api" not in model_name:
            self.model.extend_eos_tokens()

    def get_attack_mr_init_chain(self, convs_list, prompt_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Args:
            convs_list (list): List of conversation objects.
            prompt_list (list): List of prompts corresponding to each conversation.
        """

        assert len(convs_list) == len(
            prompt_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            # init_message = """{\"Round\": \"1:\" {\"improvement\": \"\",\"prompt\": \""""
            init_message = """{\"Round 1\": {\"improvement\": \""""
        else:
            init_message = """{\"Round 1\": {\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompt_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "text-davinci" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])

        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i]
                                   for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = self.model.batched_generate_by_thread(full_prompts_subset,
                                                                 max_n_tokens=self.max_n_tokens,
                                                                 temperature=self.temperature,
                                                                 top_p=self.top_p
                                                                 )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                # if gpt or vicuna-api is not add init_message
                if "gpt" not in self.model_name and "api" not in self.model_name:
                    full_output = init_message + full_output

                mr_conv, evaluation, json_str = common.extract_json_for_mr_init_chain(
                    full_output)

                if mr_conv is not None and evaluation is not None:
                    valid_outputs[orig_index] = {
                        "mr_conv": mr_conv, "evaluation": evaluation}
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If have outputs are valid, break
            if len(valid_outputs) > 0:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")

        return valid_outputs

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(
            prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "text-davinci" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])

        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i]
                                   for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                       max_n_tokens=self.max_n_tokens,
                                                       temperature=self.temperature,
                                                       top_p=self.top_p
                                                       )

            if outputs_list is None:
                print("Error in generating output.")
                indices_to_regenerate = [indices_to_regenerate[0]]
                continue

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                # if gpt or vicuna-api is not add init_message
                if "gpt" not in self.model_name and "api" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)

                # the response i not [new prompt]
                if attack_dict is not None and attack_dict["prompt"] != "[new prompt]":
                    valid_outputs[orig_index] = attack_dict
                    # Update the conversation with valid generation
                    convs_list[orig_index].update_last_message(json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if len(indices_to_regenerate) == 0:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")

        # Remove the None in valid_outputs
        valid_outputs = [
            output for output in valid_outputs if output is not None]

        return valid_outputs


class TargetLM():
    """
        Base class for target language models.

        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float,
                 preloaded_model: object = None):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, convs_list):
        full_prompts = convs_list

        retry_times = 3

        while retry_times > 0:

            try:
                retry_times -= 1
                outputs_list = self.model.batched_generate(full_prompts,
                                                           max_n_tokens=self.max_n_tokens,
                                                           temperature=self.temperature,
                                                           top_p=self.top_p
                                                           )
                break
            except Exception as e:
                print("Error at get_response: " + e)
                outputs_list = [""] * len(full_prompts)
                continue
        return outputs_list


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4", "text-davinci-003", "gpt-3.5-turbo-instruct"]:
        lm = GPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    elif model_name in ["vicuna-api", "llama2-api"]:
        lm = OpenSourceModelAPI(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def get_model_path_and_template(model_name):
    """
    TODO: Add the template you want.
    """
    full_model_dict = {
        "vicuna-api": {
            "path": "vicuna_v1.1",
            "template": "vicuna_v1.1"
        },
        "llama2-api": {
            "path": "llama-2",
            "template": "llama-2"
        },
        "one-shot": {
            "path": "one_shot",
            "template": "one_shot"
        },
        "zero-shot": {
            "path": "zero_shot",
            "template": "zero_shot"
        },
        "airoboros-1": {
            "path": "airoboros_v1",
            "template": "airoboros_v1"
        },
        "airoboros-2": {
            "path": "airoboros_v2",
            "template": "airoboros_v2"
        },
        "airoboros-3": {
            "path": "airoboros_v3",
            "template": "airoboros_v3"
        },
        "koala-1": {
            "path": "koala_v1",
            "template": "koala_v1"
        },
        "alpaca": {
            "path": "alpaca",
            "template": "alpaca"
        },
        "chatglm": {
            "path": "chatglm",
            "template": "chatglm"
        },
        "chatglm-2": {
            "path": "chatglm-2",
            "template": "chatglm-2"
        },
        "dolly-v2": {
            "path": "dolly_v2",
            "template": "dolly_v2"
        },
        "oasst-pythia": {
            "path": "oasst_pythia",
            "template": "oasst_pythia"
        },
        "oasst-llama": {
            "path": "oasst_llama",
            "template": "oasst_llama"
        },
        "tulu": {
            "path": "tulu",
            "template": "tulu"
        },
        "stablelm": {
            "path": "stablelm",
            "template": "stablelm"
        },
        "baize": {
            "path": "baize",
            "template": "baize"
        },
        "chatgpt": {
            "path": "chatgpt",
            "template": "chatgpt"
        },
        "bard": {
            "path": "bard",
            "template": "bard"
        },
        "falcon": {
            "path": "falcon",
            "template": "falcon"
        },
        "baichuan-chat": {
            "path": "baichuan_chat",
            "template": "baichuan_chat"
        },
        "baichuan2-chat": {
            "path": "baichuan2_chat",
            "template": "baichuan2_chat"
        },
        "falcon-chat": {
            "path": "falcon_chat",
            "template": "falcon_chat"
        },
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo",
            "template": "gpt-3.5-turbo"
        },
        "text-davinci-003": {
            "path": "text-davinci-003",
            "template": "text-davinci-003"
        },
        "gpt-3.5-turbo-instruct": {
            "path": "gpt-3.5-turbo-instruct",
            "template": "gpt-3.5-turbo-instruct"
        },
        "vicuna": {
            "path": "vicuna-api",
            "template": "vicuna_v1.1"
        },
        "llama2-chinese": {
            "path": "llama2_chinese",
            "template": "llama2_chinese"
        },
        "llama-2": {
            "path": "llama2-api",
            "template": "llama-2"
        },
        "claude-instant-1": {
            "path": "claude-instant-1",
            "template": "claude-instant-1"
        },
        "claude-2": {
            "path": "claude-2",
            "template": "claude-2"
        },
        "palm-2": {
            "path": "palm-2",
            "template": "palm-2"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
