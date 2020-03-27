"""API configuration class"""
import os

class Config:
    """Set configuration vars"""

    # General
    DEBUG = False

    # Database
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_USER = os.getenv('DB_USER')
    DB_PW   = os.getenv('DB_PW')
    DB_NAME = os.getenv('DB_NAME')
    DB_URL  = f'{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI        = f'postgresql://{DB_URL}'

