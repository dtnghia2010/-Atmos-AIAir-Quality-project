
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

# lstm
'''
Predict.route('/lstm-temp', methods = ['GET'])(PredictController.predictTempLSTM)
Predict.route('/lstm-humi', methods = ['GET'])(PredictController.predictHumiLSTM)
Predict.route('/lstm-co2', methods = ['GET'])(PredictController.predictCO2LSTM)
Predict.route('/lstm-co', methods = ['GET'])(PredictController.predictCOLSTM)
Predict.route('/lstm-uv', methods = ['GET'])(PredictController.predictUVLSTM)
Predict.route('/lstm-pm', methods = ['GET'])(PredictController.predictPMLSTM)
'''

# LR
Predict.route('/lr-temp', methods = ['GET'])(PredictController.predictLRTemp)
Predict.route('/lr-humi', methods = ['GET'])(PredictController.predictLRHumi)
Predict.route('/lr-co2', methods = ['GET'])(PredictController.predictLRCO2)
Predict.route('/lr-co', methods = ['GET'])(PredictController.predictLRCO)
Predict.route('/lr-uv', methods = ['GET'])(PredictController.predictLRUV)
Predict.route('/lr-pm', methods = ['GET'])(PredictController.predictLRPM)

# GB
Predict.route('/gb-temp', methods = ['GET'])(PredictController.predictGBTemp)
Predict.route('/gb-humi', methods = ['GET'])(PredictController.predictGBHumi)
Predict.route('/gb-co2', methods = ['GET'])(PredictController.predictGBCO2)
Predict.route('/gb-co', methods = ['GET'])(PredictController.predictGBCO)
Predict.route('/gb-uv', methods = ['GET'])(PredictController.predictGBUV)
Predict.route('/gb-pm', methods = ['GET'])(PredictController.predictGBPM)

# XGB
Predict.route('/xgb-temp', methods = ['GET'])(PredictController.predictXGBTemp)
Predict.route('/xgb-humi', methods = ['GET'])(PredictController.predictXGBHumi)
Predict.route('/xgb-co2', methods = ['GET'])(PredictController.predictXGBCO2)
Predict.route('/xgb-co', methods = ['GET'])(PredictController.predictXGBCO)
Predict.route('/xgb-uv', methods = ['GET'])(PredictController.predictXGBUV)
Predict.route('/xgb-pm', methods = ['GET'])(PredictController.predictXGBPM)

# RF
Predict.route('/rf-temp', methods = ['GET'])(PredictController.predictRFTemp)
Predict.route('/rf-humi', methods = ['GET'])(PredictController.predictRFHumi)
Predict.route('/rf-co2', methods = ['GET'])(PredictController.predictRFCO2)
Predict.route('/rf-co', methods = ['GET'])(PredictController.predictRFCO)
Predict.route('/rf-uv', methods = ['GET'])(PredictController.predictRFUV)
Predict.route('/rf-pm', methods = ['GET'])(PredictController.predictRFPM)

# Predict.route('/prophet-lstm', methods = ['GET'])(PredictController.predictTempProphetLSTM)

