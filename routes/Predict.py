
from flask import Blueprint

from controllers.PredictController import PredictController

Predict = Blueprint('Predict', __name__)

# prophet

Predict.route('/prophet-temp', methods = ['GET','POST'])(PredictController.predictTempProphet)
Predict.route('/prophet-humi', methods = ['GET','POST'])(PredictController.predictHumiProphet)
Predict.route('/prophet-co2', methods = ['GET','POST'])(PredictController.predictCO2Prophet)
Predict.route('/prophet-co', methods = ['GET','POST'])(PredictController.predictCOProphet)
Predict.route('/prophet-uv', methods = ['GET','POST'])(PredictController.predictUVProphet)
Predict.route('/prophet-pm', methods = ['GET','POST'])(PredictController.predictPMProphet)


# Predict.route('/prophet-lstm', methods = ['GET'])(PredictController.predictTempProphetLSTM)
# Predict.route('/gb-temp', methods = ['GET','POST'])(PredictController.predictGBTemp)

