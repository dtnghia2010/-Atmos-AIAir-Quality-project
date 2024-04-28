from flask import Blueprint

from controllers.PredictController import PredictController

Predict = Blueprint('Predict', __name__)

# Predict.route('/temp', methods = ['GET'])(PredictController.predictTempProphetLSTM)
Predict.route('/prophet', methods = ['GET'])(PredictController.predictTempProphet)
