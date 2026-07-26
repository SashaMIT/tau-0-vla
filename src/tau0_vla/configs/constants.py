"""Constants shared across the training code.

Attributes:
    IGNORE_INDEX: Label value excluded from the loss; matches the PyTorch
        ``CrossEntropyLoss`` default.
    IMAGE_TOKEN_INDEX: Token id standing in for an image (Qwen-VL).
    VIDEO_TOKEN_INDEX: Token id standing in for a video (Qwen-VL).
"""

IGNORE_INDEX = -100

# Qwen2 / Qwen2.5 / Qwen3 vocabulary (151k)
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656

# Qwen3.5 vocabulary (248k)
