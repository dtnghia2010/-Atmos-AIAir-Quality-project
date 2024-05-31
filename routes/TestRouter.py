from flask import Blueprint

from controllers.TestController import TestController

TestRouter = Blueprint('TestRouter', __name__)

TestRouter.route('/test', methods = ['GET'])(TestController.getSampleData)
TestRouter.route('/fetch', methods = ['GET'])(TestController.fetchOfflineData)
TestRouter.route('/aqi', methods = ['GET'])(TestController.calculate_aqi_from_csv)
