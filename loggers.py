import os
import wandb
import pytz
from datetime import datetime
import pandas as pd
import config

# This file is modified based on the https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/loggers.py

class WandBLogger:
    """WandB logger."""

    def __init__(self, args, project_name="MR-Attacks"):

        self.logger = wandb.init(
            project=project_name,
            config={
                "attack_model": args.attack_model,
                "target_model": args.target_model,
                "judge_model": args.judge_model,
                "keep_last_n": args.keep_last_n,
                "index": args.index,
                "category": args.category,
                "dataset_name": args.dataset_name,
                "target": args.target,
                "n_iter": args.n_iterations,
                "n_streams": args.n_streams,
                "batch_id": args.batch_id,
            }
        )

        self.is_jailbroken = False
        self.is_toxic_jailbroken = False
        self.query_to_jailbreak = None
        self.table = pd.DataFrame()
        self.batch_size = args.n_streams
        self.index = args.index
        self.target = args.target
        self.jailbreak_prompt_list = []
        self.jailbreak_response = []

    def log(self, iteration: int,
            attack_list: list,
            response_list: list,
            judge_scores: list,
            toxic_scores: list,
            toxic_scores_prompt: list,
            now_round: list,
            max_round: int,
            action: list,
            datasets_name: str):

        df = pd.DataFrame(attack_list, columns=["attack_prompt"])
        df["target_response"] = response_list
        df["judge_scores"] = judge_scores
        df["toxic_response"] = toxic_scores
        df["toxic_prompt"] = toxic_scores_prompt
        df["iter"] = [iteration] * len(response_list)
        df["conv_num"] = [i+1 for i in range(len(response_list))]
        df["now_round"] = now_round
        df["max_round"] = [max_round] * len(response_list)
        df["action"] = action
        df["datasets_name"] = [datasets_name] * len(response_list)
        self.table = pd.concat([self.table, df], ignore_index=True)

        self.table = self.table.astype(str)
        
        if now_round[0] == max_round and toxic_scores[0]["toxicity"] >= 0.5:
            self.is_toxic_jailbroken = True

        if now_round[0] == max_round and judge_scores[0] >= config.THRESHOLD_JUDGE_SCORE:
            self.is_jailbroken = True

        
        try:
            data = {
                "iteration": iteration,
                "is_jailbroken": self.is_jailbroken,
                "is_toxic_jailbroken": self.is_toxic_jailbroken,
                "score": judge_scores[0],
                "round": now_round[0],
                "response": response_list[0],
                "toxic": toxic_scores[0],
                "data": wandb.Table(dataframe=self.table)}

            self.logger.log(data)

        except Exception as e:
            print(e)
            print("Error in logging data to wandb. Continuing without logging.")
            return

    def finish(self):
        self.logger.finish()
