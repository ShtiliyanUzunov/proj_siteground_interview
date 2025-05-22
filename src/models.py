import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from PIL import Image

device = os.environ.get("DEVICE").lower()

class InstructBlipModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if not InstructBlipModelLoader._instance:
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-flan-t5-xl",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            model.eval()
            model.to(device)

            InstructBlipModelLoader._instance = (processor, model)
        return InstructBlipModelLoader._instance

def instructblip_generate_caption(image, question="What is in the image?"):
    processor, model = InstructBlipModelLoader.get_instance()
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer

class Blip2ModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if not Blip2ModelLoader._instance:
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            model.eval()
            model.to(device)

            Blip2ModelLoader._instance = (processor, model)
        return Blip2ModelLoader._instance

def blip2_generate_caption(image):
    processor, model = Blip2ModelLoader.get_instance()
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

class LlavaModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if not LlavaModelLoader._instance:
            processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True)
            model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf"
            )

            # GPU poor, can't load the model :/
            # model = model.to(device)

            LlavaModelLoader._instance = (processor, model)
        return LlavaModelLoader._instance
    
def llava_generate_caption(image):
    processor, model = LlavaModelLoader.get_instance()
    prompt = "<image>\nDescribe the image in detail."

    inputs = processor(image, prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

class MsGitModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if not MsGitModelLoader._instance:
            processor = AutoProcessor.from_pretrained("microsoft/git-base")
            model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

            model.to(device)
            MsGitModelLoader._instance = (processor, model)
        return MsGitModelLoader._instance

def ms_git_generate_caption(image):
    processor, model = MsGitModelLoader.get_instance()
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return caption


def init_caption_model():
    """
    This method ensures that the model is ready for work before the first request.
    It loads the model based on the environment variable MODEL_NAME.
    """
    model_name = os.environ.get("MODEL_NAME")
    if model_name not in ["BLIP2", "BLIPINSTRUCT", "LLAVA", "MSGIT"]:
        raise ValueError(f"Unsupported model name: {model_name}")

    loader_mapping = {
        "BLIP2": Blip2ModelLoader,
        "BLIPINSTRUCT": InstructBlipModelLoader,
        "LLAVA": LlavaModelLoader,
        "MSGIT": MsGitModelLoader
    }

    loader = loader_mapping.get(model_name)
    loader.get_instance()


def caption_fn(image):
    model_name = os.environ.get("MODEL_NAME")
    if model_name not in ["BLIP2", "BLIPINSTRUCT", "LLAVA", "MSGIT"]:
        raise ValueError(f"Unsupported model name: {model_name}")

    fn_mapping = {
        "BLIP2": blip2_generate_caption,
        "BLIPINSTRUCT": instructblip_generate_caption,
        "LLAVA": llava_generate_caption,
        "MSGIT": ms_git_generate_caption
    }

    fn = fn_mapping.get(model_name)
    return fn(image), model_name