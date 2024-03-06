import os
basedir = os.path.abspath(os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
class Config:
    # SECURITY WARNING: keep the secret key used in production secret!ß
    EMAIL_BACKEND =os.environ['EMAIL_BACKEND']
    SECRET_KEY = os.environ['SECRET_KEY']
    MAIL_SERVER = os.environ['EMAIL_HOST']
    MAIL_PORT = os.environ['EMAIL_PORT']
    MAIL_USE_TLS = os.environ['EMAIL_USE_TLS']
    SSL_REDIRECT = False
    EMAIL_HOST_USER=os.environ['EMAIL_HOST_USER']
    EMAIL_HOST_PASSWORD=os.environ['EMAIL_HOST_PASSWORD']

    @staticmethod
    def init_app(app):
        pass

