from routes.TestRouter import TestRouter
from routes.Predict import Predict

class Router:
  def run(app):
    app.register_blueprint(TestRouter, url_prefix = '/')
    app.register_blueprint(Predict, url_prefix = '/predict')