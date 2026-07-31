import argparse
import torch
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./output/checkpoint.pt")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    
    torch.save(checkpoint["model_state_dict"], "./output/model_from_checkpoint.pt")
