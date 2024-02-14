import os
import pandas as pd
import argparse
import sys

working_dir = os.path.abspath((os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")))
sys.path.append(working_dir)

from toxic_detector import ToxicDetector


dataset_path = {
    "jailbreakingLLMs": "data/harmful_behaviors_custom.csv",
    "jade": "data/jade_benchmark_en.csv",
    "gcg": "data/transfer_expriment_behaviors.csv"
}


def get_data(data_name):
    path = os.path.join(working_dir, dataset_path[data_name])
    if data_name == "jailbreakingLLMs":
        data = pd.read_csv(path)["goal"].to_list()
    elif data_name == "jade":
        data = pd.read_csv(path)["问题"].to_list()
    elif data_name == "gcg":
        # 取第一列，无column name
        data = pd.read_csv(path)["goal"].to_list()
    return data

def label_prompt(args, prompt_list):
    """
    Given a prompt, this function returns the label prompt for the prompt.
    """
    toxicd = ToxicDetector("openai")

    batch_size = args.batch_size
    
    label = []
    toxicity = []

    for i in range(0, len(prompt_list), batch_size):
        print(f"Processing {i} to {i+batch_size} prompts.")
        if i + batch_size > len(prompt_list):
            batch_size = len(prompt_list) - i

        batch_data = prompt_list[i:i+batch_size]

        toxic_scores = toxicd.get_batched_toxicity(batch_data)

        label.extend([score["type"] for score in toxic_scores])
        toxicity.extend([score["toxicity"] for score in toxic_scores])

        for prompt, toxic in zip(batch_data, toxic_scores):
            print(f"Prompt: {prompt}, Toxicity: {toxic['type']}")

    return label, toxicity

def main(args):

    prompt_list = get_data(args.data_name)

    label, toxicity = label_prompt(args, prompt_list)

    result = pd.DataFrame({"goal": prompt_list, "category": label, "toxicity": toxicity})

    result.to_csv(args.output_path, index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-name",
        default="gcg",
        help="Name of dataset.",
        required=True,
        choices=["jade", "jailbreakingLLMs", "gcg"]
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for the judge."
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the labeled data."
    )

    args = parser.parse_args()
    main(args)

