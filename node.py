# Use a pipeline as a high-level helper
import os.path
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        default_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "banned-detect-image.jpg") #获取默认图片路径
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "nsfw_threshold"}),
                # "alternative_image": ("IMAGE",{"default": default_image_path}),
            },
            "optional": {
                "alternative_image": ("IMAGE", {"default": default_image_path}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

