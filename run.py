import argparse
import sys
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="HEMS FDI Detection")

    parser.add_argument(
        "--model",
        type=str,
        default="sac",
        choices=["sac", "gat", "td3", "ddpg", "preprocess"],
        help="Model to run or 'preprocess' to run data prep."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run mode"
    )

    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    
    # We pass along the unknown args to the sub-scripts if needed
    
    # Set PYTHONPATH to root of project
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    if args.model == "sac":
        if args.mode == "train":
            subprocess.run([sys.executable, "src/agents/lstm_sac_main.py"] + unknown, env=env)
        else:
            subprocess.run([sys.executable, "src/agents/test_lstm_sac.py"] + unknown, env=env)

    elif args.model == "gat":
        if args.mode == "train":
            subprocess.run([sys.executable, "src/gat_model/main.py"] + unknown, env=env)
        else:
            subprocess.run([sys.executable, "src/gat_model/test_GAT_LSTM.py"] + unknown, env=env)
            
    elif args.model == "td3":
        if args.mode == "train":
            subprocess.run([sys.executable, "src/agents/lstm_td3_main.py"] + unknown, env=env)
        else:
            print("Testing td3 is not yet implemented.")

    elif args.model == "ddpg":
        if args.mode == "train":
            subprocess.run([sys.executable, "src/agents/lstm_ddpg_main.py"] + unknown, env=env)
        else:
            print("Testing ddpg is not yet implemented.")
            
    elif args.model == "preprocess":
        print("Running data preprocessing...")
        subprocess.run([sys.executable, "src/data_processing/Data_Preprocessing.py"] + unknown, env=env)
        print("Generating attacked data...")
        subprocess.run([sys.executable, "src/data_processing/make_attacked_data.py"] + unknown, env=env)
        print("Data setup complete.")

if __name__ == "__main__":
    main()