import yaml
import argparse
from utils import build_model

def count_params(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Build the model using your utility function
    model = build_model(config=config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Configuration: {config_path}")
    print(f"Total Parameters: {num_params:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters of a model from a config file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config yaml file")
    
    args = parser.parse_args()
    count_params(args.config_path)