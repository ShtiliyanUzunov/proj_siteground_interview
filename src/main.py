from dotenv import load_dotenv

load_dotenv()

import logging
from logging_config import log_service_config, setup_logging
from models import init_caption_model

import os
from flask import Flask
from routes import api_bp

app = Flask(__name__)
app.register_blueprint(api_bp)
setup_logging()


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    log_service_config()
    init_caption_model()
    
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port)