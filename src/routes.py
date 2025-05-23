import logging
import os
import time
from flask import Blueprint, request, jsonify
from PIL import Image

from tasks import TaskManager, TaskQueueFullError, TaskStatus
from utils import resize_to_max_dim
from models import caption_fn

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)

@api_bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


@api_bp.route("/caption", methods=["POST"])
def caption():
    """
    Directly captions the image and returns the result.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(image_file.stream).convert("RGB")
        image = resize_to_max_dim(image, int(os.getenv("MAX_IMAGE_SIZE", 1024)))

        generated_caption, model_name, processing_time = caption_fn(image)
        
        logger.info(f"Image processed in {processing_time:.2f} seconds using model: {model_name}")

        return jsonify({
            "caption": generated_caption,
            "processing_time": processing_time,
            "model_name": model_name
        }), 200

    except Exception as exc:
        logger.error(f"Error processing image: {exc}")
        return jsonify({"error": "Something went wrong on our side..."}), 500
    
@api_bp.route("/caption-task", methods=["POST"])
def caption_task():
    """
    Creates a task for captioning the image and returns the task ID.
    The task state can be checked later using the task ID.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(image_file.stream).convert("RGB")
        image = resize_to_max_dim(image, int(os.getenv("MAX_IMAGE_SIZE", 1024)))

        task_id = TaskManager.get_instance().add_task(image)

        return jsonify({
            "task_id": task_id
        }), 200
    
    except TaskQueueFullError:
        return jsonify({"error": "The task queue is currently full. Try again later."}), 429

    except Exception as exc:
        logger.error(f"Error processing image: {exc}")
        return jsonify({"error": "Something went wrong on our side..."}), 500
    
@api_bp.route("/caption-task/<task_id>", methods=["GET"])
def get_caption_task(task_id):
    task = TaskManager.get_instance().get_task(task_id)

    if task is None:
        return jsonify({"error": "Task not found"}), 404

    response = {
        "task_id": task_id,
        "status": task["status"]
    }

    if task["status"] == TaskStatus.COMPLETED:
        response["result"] = task["result"]
    elif task["status"] == TaskStatus.FAILED:
        response["error"] = task["error"]

    return jsonify(response), 200

@api_bp.route("/caption-task", methods=["GET"])
def get_all_caption_tasks():
    manager = TaskManager.get_instance()
    with manager.lock:  # ensure thread-safe read
        all_tasks = {
            task_id: {
                "status": task["status"],
                "result": task.get("result"),
                "error": task.get("error")
            }
            for task_id, task in manager.tasks.items()
        }

    return jsonify(all_tasks), 200