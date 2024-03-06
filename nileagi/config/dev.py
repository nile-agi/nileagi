from config.base import Config
import os
basedir = os.path.abspath(os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv()

class DevConfig(Config):
    DB_HOST = os.getenv('DB_HOST_DEV')
    DB_NAME = os.getenv('DB_NAME_DEV')
    DB_USER = os.getenv('DB_USER_DEV')
    DB_PASSWORD = os.getenv('DB_PASSWORD_DEV')
    DB_PORT = os.getenv('DB_PORT_DEV')
    RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
    REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
    FIREWORKS_API_KEY = os.getenv('FIREWORKS_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    ALLOWED_TEST_USERS = os.getenv('ALLOWED_TEST_USERS')
