import os
import torch
from transformers import AutoModelForCausalLM
import typer

def main(src: str, tgt: str):
    model = AutoModelForCausalLM.from_pretrained(src, device_map="cpu")
    torch.save(model.state_dict(), os.path.join(tgt, "model.pt"))

if __name__ == "__main__":
    typer.run(main)
