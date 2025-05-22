import logging
import os
import time
from flask import Blueprint, request, jsonify
from PIL import Image

from utils import resize_to_max_dim
from models import caption_fn

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)

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
        image = resize_to_max_dim(image, int(os.getenv("MAX_IMAGE_SIZE", 1024)))

        start = time.time()
        generated_caption, model_name = caption_fn(image)
        processing_time = time.time() - start
        
        logger.info(f"Image processed in {processing_time:.2f} seconds using model: {model_name}")

        return jsonify({
            "caption": generated_caption,
            "processing_time": processing_time,
            "model_name": model_name
        }), 200

    except Exception as exc:
        logger.error(f"Error processing image: {exc}")
        return jsonify({"error": "Something went wrong on our side..."}), 500