import os
import wandb
import pytz
from datetime import datetime
import pandas as pd
import config
import random
from round_manager import Task
import json

# This file is modified based on the https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/loggers.py

class WandBLogger:
    """WandB logger."""

    def __init__(self, args, init_chain, project_name="MR-Attacks"):

        self.logger = wandb.init(
            project=project_name,
            # name 8 bit random string by default
            reinit=True,
            name=''.join(random.choices('0123456789ABCDEF', k=8)),
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
                "init_chain": init_chain,
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

    def log(self, task: Task):

        df = pd.DataFrame(task.get_prompts(), columns=["attack_prompt"])
        df["target_response"] = task.get_responses()
        df["judge_scores"] = task.get_judge_scores()
        df["toxic_response"] = task.get_response_toxic()
        df["toxic_prompt"] = task.get_prompt_toxic()
        df["iter"] = task.get_now_index()
        df["conv_num"] = [i+1 for i in range(len(task.get_prompts()))]
        df["now_round"] = task.get_now_rounds()
        df["max_round"] = task.get_max_rounds()
        df["action"] = [action.name for action in task.get_actions()]
        df["datasets_name"] = task.get_datasets_name()
        df["sem_prompt"] = task.get_prompt_sem()
        df["sem_response"] = task.get_response_sem()
        self.table = pd.concat([self.table, df], ignore_index=True)

        if task.get_now_rounds()[0] == task.get_max_rounds()[0] and task.get_response_toxic()[0] >= 0.5:
            self.is_toxic_jailbroken = True
        
        if task.get_now_rounds()[0] == task.get_max_rounds()[0] and task.get_judge_scores()[0] >= config.THRESHOLD_JUDGE_SCORE:
            self.is_jailbroken = True
        
        # Attack chain prompt
        attack_chain_prompt = []
        attack_chain_response = []
        
        for i in range(len(self.table)):
            attack_chain_prompt.append({"prompt": self.table["attack_prompt"][i], "round": self.table["now_round"][i], "action": self.table["action"][i]})
            attack_chain_response.append({"response": self.table["target_response"][i], "round": self.table["now_round"][i], "action": self.table["action"][i]})

        try:
            data = {
                "iteration": task.get_now_index()[0],
                "is_jailbroken": self.is_jailbroken,
                "is_toxic_jailbroken": self.is_toxic_jailbroken,
                "judge": task.get_judge_scores()[0],
                "round": task.get_now_rounds()[0],
                "prompt": task.get_prompts()[0],
                "response": task.get_responses()[0],
                "toxic_response": task.get_response_toxic()[0],
                "sem_response": task.get_response_sem()[0],
                "sem_prompt": task.get_prompt_sem()[0],
                "toxic_prompt": task.get_prompt_toxic()[0],
                "attack_chain_prompt": attack_chain_prompt,
                "attack_chain_response": attack_chain_response,
                "data": wandb.Table(dataframe=self.table)}

            self.logger.log(data)

        except Exception as e:
            print(e)
            print("Error in logging data to wandb. Continuing without logging.")
            return

    def finish(self):
        self.logger.finish()

        print("Logging finished.")

class LocalLogger:
    """Local Logger."""

    def __init__(self, args, init_chain, project_name="MR-Attacks"):

        self.logger = {
            "project": project_name,
            "name": ''.join(random.choices('0123456789ABCDEF', k=8)),
            "config": {
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
                "init_chain": init_chain,
            },
            "start_time": datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        }
        print("Local logger: ", self.logger.get("name"))
        
        self.is_jailbroken = False
        self.is_toxic_jailbroken = False
        self.query_to_jailbreak = None
        self.table = {}
        self.batch_size = args.n_streams
        self.index = args.index
        self.target = args.target
        self.jailbreak_prompt_list = []
        self.jailbreak_response = []
        self.attack_chain_result = {}

        # save to local
        local_path = os.path.join("logs", self.logger["project"], self.logger["name"])
        self.local_path = local_path
        
        os.makedirs(local_path, exist_ok=True)

        # save logs to json
        with open(os.path.join(self.local_path, "logs.json"), "w") as f:
            json.dump(self.logger, f)


    def savejson(self, data, filename):
        with open(os.path.join(self.local_path, filename), "w") as f:
            json.dump(data, f)

    def log(self, task: Task):
        result = {}
        result["attack_prompt"] = task.get_prompts()
        result["target_response"] = task.get_responses()
        result["judge_scores"] = task.get_judge_scores()
        result["toxic_response"] = task.get_response_toxic()
        result["toxic_prompt"] = task.get_prompt_toxic()
        result["iter"] = task.get_now_index()
        result["now_round"] = task.get_now_rounds()
        result["max_round"] = task.get_max_rounds()
        result["action"] = [action.name for action in task.get_actions()]
        result["datasets_name"] = task.get_datasets_name()
        result["sem_prompt"] = task.get_prompt_sem()
        result["sem_response"] = task.get_response_sem()
        result["target"] = self.target

        self.table[str(result["iter"][0])] = result

        if task.get_response_toxic()[0] >= 0.5:
            self.is_toxic_jailbroken = True

        for batch in range(self.batch_size):
            if task.get_judge_scores()[batch] == 1:
                self.is_jailbroken = True
                # Get the attack chain
                for i in range(1, len(self.table) + 1):
                    self.attack_chain_result[str(i)] = {
                        "attack_prompt": self.table[str(i)]["attack_prompt"][batch],
                        "target_response": self.table[str(i)]["target_response"][batch],
                        "now_round": self.table[str(i)]["now_round"][batch],
                        "action": self.table[str(i)]["action"][batch],
                        "toxic_prompt": self.table[str(i)]["toxic_prompt"][batch],
                        "toxic_response": self.table[str(i)]["toxic_response"][batch],
                        "sem_prompt": self.table[str(i)]["sem_prompt"][batch],
                        "sem_response": self.table[str(i)]["sem_response"][batch],
                        "judge_scores": self.table[str(i)]["judge_scores"][batch],
                        "target": self.target,
                        "datasets_name": self.table[str(i)]["datasets_name"][batch],
                        "batch": batch,
                    }
                break

        try:
            # save to local as csv
            table_path = os.path.join(self.local_path, "table")
            os.makedirs(table_path, exist_ok=True)

            # save log to json
            self.savejson(self.table, os.path.join("table", "table-" + str(task.get_now_index()[0]) + ".json"))

        except Exception as e:
            print(e)
            print("Error in logging data to wandb. Continuing without logging.")
            return


    def finish(self):
        # save table to csv
        self.logger["is_jailbroken"] = self.is_jailbroken
        if self.is_jailbroken:
            self.logger["final_response"] = self.attack_chain_result[str(len(self.attack_chain_result))]["target_response"]
            self.logger["final_prompt"] = self.attack_chain_result[str(len(self.attack_chain_result))]["attack_prompt"]
        else:
            self.logger["final_response"] = ""
            self.logger["final_prompt"] = ""
        self.logger["end_time"] = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        self.logger["time_cost"] = (datetime.strptime(self.logger["end_time"], "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.logger["start_time"], "%Y-%m-%d %H:%M:%S")).seconds

        # resave the config
        with open(os.path.join(self.local_path, "logs.json"), "w") as f:
            json.dump(self.logger, f)

        if self.is_jailbroken:
            self.savejson(self.attack_chain_result, "passed.json")
        else:
            self.attack_chain_result["result"] = "failed"
            self.savejson(self.attack_chain_result, "failed.json")

        print("Logging finished.")
        