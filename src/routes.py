import os
import time
from flask import Blueprint, request, jsonify
from PIL import Image

from utils import resize_to_max_dim
from models import instructblip_generate_caption
from models import llava_generate_caption
from models import ms_git_generate_caption
from models import blip2_generate_caption

api_bp = Blueprint("api", __name__)

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

@api_bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


@api_bp.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(image_file.stream).convert("RGB")
        image = resize_to_max_dim(image, 1024)

        start = time.time()
        generated_caption, model_name = caption_fn(image)
        processing_time = time.time() - start

        return jsonify({
            "caption": generated_caption,
            "processing_time": processing_time,
            "model_name": model_name
        }), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500