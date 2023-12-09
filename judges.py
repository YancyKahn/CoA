from fastchat.model import (
    get_conversation_template
)
import re
from system_prompts import get_judge_system_prompt

from language_models import GPT, OpenSourceModelAPI


# This file is modified based on the https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/judges.py

def load_judge(args):
    if "gpt" in args.judge_model or "text-davinci" in args.judge_model:
        return GPTJudge(args)
    elif args.judge_model == "no-judge":
        return NoJudge(args)
    elif "api" in args.judge_model:
        return OpenSourceJudge(args)
    else:
        raise NotImplementedError


class LocalJudge:
    def __init__(self):
        pass

    def score(self, round_list, max_round, target, prompt_list, response_list):
        pass


class JudgeBase:
    def __init__(self, args):
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name = args.judge_model
        self.target = args.target
        self.system_prompt = get_judge_system_prompt(args.target)

    def get_judge_prompt(self, round, max_round, target, attack_prompt, target_response):
        return f"[Round]: {round}/{max_round}\n [TARGET]:{target}\n[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"

    def score(self, prompt_list, response_list):
        raise NotImplementedError

    def process_output(self, raw_output):
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            print(f"Error in processing judge output: {raw_output}")
            output = 1
        return output


class NoJudge(JudgeBase):
    def __init__(self, args):
        super(NoJudge, self).__init__(args)
        self.judge_model = None

    def score(self, round_list, max_round, target, prompt_list, response_list):
        return [1 for _ in prompt_list]


class GPTJudge(JudgeBase):
    def __init__(self, args):
        super(GPTJudge, self).__init__(args)
        self.judge_model = GPT(model_name=self.judge_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def score(self, round_list, max_round, target, attack_prompt_list, target_response_list):
        max_round_list = [max_round for _ in round_list]
        target_list = [target for _ in round_list]
        convs_list = []

        for round, max_round, target, attack_prompt, target_response in zip(round_list, max_round_list, target_list, attack_prompt_list, target_response_list):
            convs_list.append(self.create_conv(self.get_judge_prompt(
                round, max_round, target, attack_prompt, target_response)))

        raw_outputs = self.judge_model.batched_generate(convs_list,
                                                        max_n_tokens=self.max_n_tokens,
                                                        temperature=self.temperature)
        outputs = [self.process_output(raw_output)
                   for raw_output in raw_outputs]
        return outputs


class OpenSourceJudge(JudgeBase):
    def __init__(self, args):
        super(OpenSourceJudge, self).__init__(args)
        self.args = args
        self.judge_model = OpenSourceModelAPI(model_name=args.judge_model)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.get_prompt()

    def score(self, round_list, max_round, target, attack_prompt_list, target_response_list):
        max_round_list = [max_round for _ in round_list]
        target_list = [target for _ in round_list]
        convs_list = []

        for round, max_round, target, attack_prompt, target_response in zip(round_list, max_round_list, target_list, attack_prompt_list, target_response_list):
            convs_list.append(self.create_conv(self.get_judge_prompt(
                round, max_round, target, attack_prompt, target_response)))

        raw_outputs = self.judge_model.batched_generate(convs_list,
                                                        max_n_tokens=self.max_n_tokens,
                                                        temperature=self.temperature,
                                                        top_p=0.9)
        outputs = [self.process_output(raw_output)
                   for raw_output in raw_outputs]
        return outputs
