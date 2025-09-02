import logging

import torch


def get_device_type() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        logging.warning("No GPU found, using CPU instead")
        return "cpu"


device_type = get_device_type()
DEVICE = torch.device(device_type)
