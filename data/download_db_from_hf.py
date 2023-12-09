from datasets import load_dataset
import argparse
import os

# download the dataset from HuggingFace
def download_db(dataset_name, save_path):
    dataset = load_dataset(dataset_name)

    if save_path is None:
        # The File path
        working_filedir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(working_filedir, dataset_name.split("/")[1] + ".csv")
    # save format is csv
    dataset["train"].to_csv(save_path, index=False, header=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--dataset-name", 
                      type=str, 
                      default="PKU-Alignment/PKU-SafeRLHF",
                      help="Name of the dataset to download from HuggingFace.")
    
    args.add_argument("--save-path",
                        type=str,
                        default=None,
                        help="Path to save the dataset.")
    
    args = args.parse_args()

    download_db(args.dataset_name, args.save_path) 