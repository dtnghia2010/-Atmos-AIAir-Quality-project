from flask import Blueprint

from controllers.PredictController import PredictController

Predict = Blueprint('Predict', __name__)

Predict.route('/prophet', methods = ['GET'])(PredictController.predictTempProphet)
# Predict.route('/prophet-lstm', methods = ['POST'])(PredictController.predictTempProphetLSTM)
