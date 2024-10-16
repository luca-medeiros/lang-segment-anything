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
