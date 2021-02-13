from flask import *


app = Flask(__name__)
app.secret_key = 'random string'

if __name__ == '__main__':
    app.run(debug=False)
