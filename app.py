from flask import Flask
from flask import render_template
from flask import request
import utils.SQLiteDB as dbHandler
import utils.preprocessing


app = Flask(__name__)



@app.route("/", methods=['POST', 'GET'])
def home():
    dbHandler.createTableIfNotExist()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        dbHandler.insertUser(username, password)
        users = dbHandler.retrieveUsers()
        return render_template('index.html', users=users)
    else:
        return render_template('index.html')


@app.route("/user")
def users():
    return dbHandler.retrieveUsers()

@app.route("/register", methods=['POST', 'GET'])
def registerUsers():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        return dbHandler.registerUsers(username, email, password)
    else:
        return render_template('signup.html')


@app.route("/userWithUsername", methods=['POST', 'GET'])
def userWithUsername():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
    return dbHandler.retrieveUsersWithUsername(username)


@app.route("/predict", methods=['POST'])
def prediction():
    data = request.get_data()
    print(data)
    # print('Vectorized Input:')
    pred = utils.preprocessing.getPrediction(data)

    if pred[0] == 0:
        print("It seems to be safe input")
        dbresponse = dbHandler.executeQuery(data.decode())
        print("DB Response =", dbresponse)
        return "It seems to be safe input"
    else:
        print("ALERT :::: This can be SQL injection")
        return "ALERT :::: This can be SQL injection"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
