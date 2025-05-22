# Usage:
# python -m locust -f .\test\locust.py --headless -u 10 -r 10
# -u defines the number of users
# -r defines the spawn rate
# -u 10 -r 10 means 10 users will be spawned at a rate of 10 users per second (The test starts with all 10 users at once)

from locust import HttpUser, task, between, events
import threading

TOTAL_REQUESTS = 50
request_counter = 0
failure_counter = 0
response_times = []
lock = threading.Lock()

class CaptionUser(HttpUser):
    wait_time = between(0, 0)
    host = "http://localhost:5000"

    @task
    def post_image_caption(self):
        global request_counter, failure_counter

        with lock:
            if request_counter >= TOTAL_REQUESTS:
                self.environment.runner.quit()
                return
            request_counter += 1
            current = request_counter

        with open("notebooks/images/basketball.jpg", "rb") as img:
            files = {"image": ("test.jpg", img, "image/jpeg")}
            with self.client.post("/caption", files=files, catch_response=True) as response:
                response_times.append(response.json()["processing_time"])
                if response.status_code != 200:
                    failure_counter += 1
                    response.failure(f"Status: {response.status_code} - {response.text}")
                else:
                    response.success()

        if current >= TOTAL_REQUESTS:
            self.environment.runner.quit()

@events.quitting.add_listener
def _(environment, **kwargs):
    log_file = "test/performance.log"
    users = getattr(environment.parsed_options, "num_users", "N/A")  
    spawn_rate = getattr(environment.parsed_options, "spawn_rate", "N/A")  

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("========== Test Summary ==========\n")
        f.write(f"Total users        : {users}\n")
        f.write(f"Spawn rate         : {spawn_rate}\n")
        f.write(f"Total requests sent: {request_counter}\n")
        f.write(f"Total failures     : {failure_counter}\n")

        if response_times:
            avg_time = sum(response_times) / len(response_times)
            f.write(f"Average response time: {avg_time:.3f} seconds\n")
        else:
            f.write("No response times recorded.\n")

        f.write("==================================\n")
        f.flush()

