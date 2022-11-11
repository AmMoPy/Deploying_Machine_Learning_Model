import os
from dotenv import load_dotenv

# locating and loading .env file containing SECRET_KEY
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'env_v.env'))

upload_folder = os.path.dirname(os.path.abspath(__file__))

# configs
UPLOAD_FOLDER = upload_folder
UPLOAD_EXTENSIONS = ['.json']

# pulling secret key from .env file
# or use any key of your choice
SECRET_KEY = os.environ.get('SECRET_KEY')