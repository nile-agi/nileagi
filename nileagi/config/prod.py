from config.base import Config
import os
basedir = os.path.abspath(os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv()

class ProdConfig(Config):
    DB_HOST = os.getenv('DB_HOST_PROD')
    DB_NAME = os.getenv('DB_NAME_PROD')
    DB_USER = os.getenv('DB_USER_PROD')
    DB_PASSWORD = os.getenv('DB_PASSWORD_PROD')
    DB_PORT = os.getenv('DB_PORT_PROD')
    RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
    REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
    FIREWORKS_API_KEY = os.getenv('FIREWORKS_API_KEY')