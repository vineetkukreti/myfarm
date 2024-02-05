
from flask import Flask, render_template, request, session, redirect, url_for,jsonify
import pickle
import os
import numpy as np
import os
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

# import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

model = pickle.load(open('saved_models\\NBClassifier.pkl', 'rb'))

from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
  
app = Flask(__name__)
  
  
app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user-system'
  
mysql = MySQL(app)
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('layout.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)


  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return render_template('index.html')
  
@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)', (userName, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)

@app.route('/predict', methods=['POST'])    
def predict():
    if request.method == 'POST':
        N = request.form.get('N')
        P = request.form.get('P')
        K = request.form.get('K')
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')
        ph = request.form.get('ph')
        rainfall = request.form.get('rainfall')

        # Assuming N, P, K, temperature, humidity, ph, and rainfall are defined elsewhere
        input_data = [float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]

        # Perform prediction using the loaded Naive Bayes model
        prediction = model.predict([input_data])[0]

        # You can use the prediction result in your template or further processing
        return render_template('prediction.html', s=prediction)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/p')
def p():
    return render_template('prediction.html')



@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/pages')
def pages():
    return render_template('pages.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/detail')
def detail():
    return render_template('detail.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')

@app.route('/team')
def team():
    return render_template('team.html')


best_model_inceptionresnetv2 = load_model("saved_models\\inceptionresnetv2.h5")

# Class labels
class_labels = ['Africanized Honey Bees (Killer Bees)',
                'Aphids',
                'Armyworms',
                'Brown Marmorated Stink Bugs',
                'Cabbage Loopers',
                'Citrus Canker',
                'Colorado Potato Beetles',
                'Corn Borers',
                'Corn Earworms',
                'Fall Armyworms',
                'Fruit Flies',
                'Spider Mites',
                'Thrips',
                'Tomato Hornworms',
                'Western Corn Rootworms']

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Get the uploaded file
        print(request.files)
        file = request.files['file']
        
        # Save the file to a temporary location
        file.save('static/temp_image.jpg')

        # Load and preprocess the image
        img_path = 'static/temp_image.jpg'
        img = image.load_img(img_path, target_size=(256, 256,3))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = best_model_inceptionresnetv2.predict(img_array)

        # Get the top predicted class index
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        return render_template('dangerous_insects.html', prediction=predicted_class_label, img_path=img_path)

    # Handle GET request (show the form)
    return render_template('dangerous_insects.html')  # Adjust the template name as needed


#---------------------------------------------------------- This is for the fish name predictor -----------------------------------------------
# Load the scaler model
with open('saved_models\\fish_farm\\scaler_model.joblib', 'rb') as scaler_file:
    scaler_model = joblib.load(scaler_file)

# Load the DecisionTreeClassifier model
with open('saved_models\\fish_farm\\best_dt_model.joblib', 'rb') as dt_model_file:
    best_dt_model = joblib.load(dt_model_file)



@app.route('/predict_fish', methods=['POST', 'GET'])
def predict_fish():
    if request.method == 'POST':
        # Get input data from the request form
        ph = float(request.form['ph'])
        temperature = float(request.form['temperature'])
        turbidity = float(request.form['turbidity'])

        # Preprocess features (you may need to adjust this based on your original scaling)
        features = scaler_model.transform([[ph, temperature, turbidity]])

        # Make prediction
        prediction = best_dt_model.predict(features)

        # Render the template with the prediction result
        return render_template('fish_farm.html', prediction=prediction[0])
    else:
        # Render the template for the initial GET request
        return render_template('fish_farm.html')
    
if __name__=='__main__':
    app.run(debug=True)