# utils.py

import os
import torch

def save_checkpoint(model, optimizer, episode, save_path):
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode
    }
    torch.save(checkpoint, os.path.join(save_path, f'checkpoint_{episode}.pth'))
