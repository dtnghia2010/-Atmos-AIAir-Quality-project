from flask import Flask
# from config import HOST, PORT, DEBUG
from routes.Router import Router
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Rest API with Python Flask</h1>'

Router.run(app)

if __name__ == '__main__':
    app.run(debug=True)