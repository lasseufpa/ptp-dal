import threading, time, logging
from flask import Flask, Blueprint
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from resources import dataset
from models import db
from manage import Watcher

def get_logger(app):
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def create_app():
    # Create flask object
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # Create API blueprint
    api_bp = Blueprint('api', __name__)
    api    = Api(api_bp)

    # Create API endpoints
    api.add_resource(dataset.DatasetDownload, '/dataset/<dataset>')
    api.add_resource(dataset.DatasetSearch, '/search')
    app.register_blueprint(api_bp, url_prefix='/api')

    # Init database
    time.sleep(10) # Wait a while for the database to be ready
    db.init_app(app)
    with app.app_context():
        db.drop_all()
        db.create_all()

    return app


# Create app instance
app = create_app()

# Configure application logger
get_logger(app)

# Init file system watcher
watcher  = Watcher(app)
watching = threading.Thread(target=watcher.run)
watching.start()
