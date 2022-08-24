import logging
import os

import flask
from flask import jsonify, request

from classifier.search_text import wrapper

logging.basicConfig(level=logging.INFO)
__version__ = "1.0.0"
logger = logging.getLogger(__name__)


def default_error_handler(error):
    return {'message': str(error), 'version': __version__}, getattr(error, 'code', 500)


def get_status():
    return jsonify(dict(message='ok', version=__version__))


def classify():
    return jsonify(wrapper(request.json))


def create_app():
    app = flask.Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.register_error_handler(Exception, default_error_handler)
    app.add_url_rule('/api/status', view_func=get_status, methods=['GET'])
    app.add_url_rule('/api/classify', view_func=classify, methods=['POST'])
    return app


def env_var(vname, default_val=None):
    if vname not in os.environ:
        msg = f'define {vname} environment variable! defaulting to {default_val}'
        logger.warning(msg)
        return default_val
    else:
        return os.environ[vname]


if __name__ == "__main__":
    port = env_var('DOC_CLASSIFIER_SERVICE_PORT', 5002)
    create_app().run(host='0.0.0.0', port=port)
