import logging
import os
import threading
import time
import uuid
from typing import Dict, Optional
from threading import Lock

from models import caption_fn

logger = logging.getLogger(__name__)

"""
Primitives for task management.
Ideally the tasks should be stored externally, 
so that when the service is restarted the task state is preserved, and the task runner can continue the execution.
For simplicity an in-memory dictionary is used.
"""


class TaskStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskQueueFullError(RuntimeError):
    pass


class TaskManager:
    _instance = None

    def __init__(self, max_queue_size: int):
        self.max_queue_size = max_queue_size
        self.tasks: Dict[str, Dict] = {}
        self.lock = Lock()

    @staticmethod
    def get_instance():
        if TaskManager._instance is None:
            TaskManager._instance = TaskManager(
                max_queue_size=int(os.getenv("TASK_QUEUE_SIZE", 5))
            )
        return TaskManager._instance

    def add_task(self, image) -> str:
        with self.lock:
            if self._pending_task_count() >= self.max_queue_size:
                raise TaskQueueFullError("Task queue is full")

            task_id = str(uuid.uuid4())
            self.tasks[task_id] = {
                "status": TaskStatus.PENDING,
                "image": image,
                "result": None,
                "error": None,
            }
            return task_id

    def _pending_task_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t["status"] == TaskStatus.PENDING)

    def get_task(self, task_id: str) -> Optional[Dict]:
        with self.lock:
            return self.tasks.get(task_id)

    def set_status(self, task_id: str, status: str):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = status

    def set_result(self, task_id: str, result: str):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = TaskStatus.COMPLETED
                self.tasks[task_id]["result"] = result

    def set_error(self, task_id: str, error: str):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = TaskStatus.FAILED
                self.tasks[task_id]["error"] = error


class TaskProcessor(threading.Thread):
    _instance = None

    def __init__(self, poll_interval: float = 0.1):
        super().__init__(daemon=True)
        self.task_manager = TaskManager.get_instance()
        self.caption_fn = caption_fn
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()

    @staticmethod
    def get_instance():
        if TaskProcessor._instance is None:
            TaskProcessor._instance = TaskProcessor()
        return TaskProcessor._instance

    def run(self):
        while not self._stop_event.is_set():
            pending_task = self._get_next_pending_task()
            if pending_task:
                logger.info(f"Processing task {pending_task[0]}")
                task_id, task_data = pending_task
                try:
                    self.task_manager.set_status(task_id, TaskStatus.PROCESSING)
                    caption, model_name, processing_time = self.caption_fn(
                        task_data["image"]
                    )
                    self.task_manager.set_result(
                        task_id,
                        {
                            "caption": caption,
                            "model": model_name,
                            "processing_time": processing_time,
                        },
                    )
                except Exception as e:
                    self.task_manager.set_error(task_id, str(e))
            else:
                time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()

    def _get_next_pending_task(self):
        with self.task_manager.lock:
            for task_id, task in self.task_manager.tasks.items():
                if task["status"] == TaskStatus.PENDING:
                    return task_id, task
        return None
