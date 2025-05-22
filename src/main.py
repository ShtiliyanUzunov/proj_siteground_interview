import logging
from dotenv import load_dotenv

from logging_config import log_service_config, setup_logging
load_dotenv()

import os
from flask import Flask
from routes import api_bp

app = Flask(__name__)
app.register_blueprint(api_bp)
setup_logging()


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    log_service_config()
    
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host=host, port=port)