from flask import Flask,redirect,render_template,request,flash,url_for,send_from_directory,session,jsonify
from webforms import LoginForm,RegistrationForm,DiseaseField,ContactForm
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import UserMixin, login_user, LoginManager,login_required, logout_user, current_user
from flask_mail import Mail, Message
from datetime import datetime
import os
import uuid

import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Old SQLite DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farm.db'

# Initialize The Database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

UPLOAD_FOLDER = 'uploads'

# Secret Key!
app.config['SECRET_KEY'] = "my super secret key that no one is supposed to know"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'nakuldesai2006@gmail.com'  # Use your actual Gmail address
app.config['MAIL_PASSWORD'] = 'jcfd zcoa bajh zpap'     # Use your generated App Password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx', 'xlsx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



# Flask Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
app.config['LOGIN_VIEW'] = 'login'



# Define the class names based on your model
class_names = ['Apple___Apple_scab',
               'Apple___Black_rot',
               'Apple___Cedar_apple_rust',
               'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Raspberry___healthy',
               'Soybean___healthy',
               'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch',
               'Strawberry___healthy',
               'Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


def ImageProcessing(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    
    if img is None:
        print("Error: Image not found or cannot be read.")
        return
    
    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the required size (224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalize the image (rescale pixel values to [0, 1])
    img_normalized = img_resized / 255.0
    
    # Convert the image to a batch format
    input_arr = np.array([img_normalized])

    model = tf.keras.models.load_model('plant_disease_detection.h5') # type: ignore
    
    # Perform prediction
    prediction = model.predict(input_arr)
    
    # Get the index of the highest probability class
    result_index = np.argmax(prediction)
    
    # Retrieve the class name corresponding to the highest probability
    model_prediction = class_names[result_index]
    
    return model_prediction



def allowed_file(filename):
    """
    Check if the uploaded file is allowed based on its extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create Model
class Users(db.Model,UserMixin):
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(20), nullable=True, unique=True)
	name = db.Column(db.String(200), nullable=False)
	email = db.Column(db.String(120), nullable=False, unique=True)
	# Do some password stuff!
	password_hash = db.Column(db.String(200),nullable=True)


	@property
	def password(self):
		raise AttributeError('password is not a readable attribute!')

	@password.setter
	def password(self, password):
		self.password_hash = generate_password_hash(password)

	def verify_password(self, password):
		return check_password_hash(self.password_hash, password)

	# Create A String
	def __repr__(self):
		return '<Name %r>' % self.name
     
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True, auto_increment=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
	

@login_manager.user_loader
def load_user(user_id):
	return Users.query.get(int(user_id))

@app.route('/')
def home():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('homeGuj'))
    session['language'] = "English"
    return render_template('index.html')

@app.route('/about')
def about():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('aboutGuj'))
    return render_template('about.html')

@app.route('/explore')
def explore():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('exploreGuj'))
    return render_template('explore.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
        loginform = LoginForm()
        signupform = RegistrationForm()

        if loginform.validate_on_submit():
            user = Users.query.filter_by(username=loginform.username.data).first()
            if user:
                # Check the hash
                if check_password_hash(user.password_hash, loginform.password.data):
                    login_user(user)
                    flash("Login Success")
                    return redirect(url_for('home'))
                else:
                    flash("Wrong Credentials - Try Again!")
            else:
                flash("User Does Not Exist!")
        return render_template('login.html', loginform=loginform, signupform=signupform)

@app.route('/register', methods=['GET', 'POST'])
def register():
    signupform = RegistrationForm()
    loginform = LoginForm()
    if signupform.validate_on_submit():
        username = signupform.username.data
        if Users.query.filter_by(username=username).first():
            flash("Username already exists. Please choose a different username.")
            return redirect(url_for('login'))
        hashed_password = generate_password_hash(signupform.password.data)
        new_user = Users(name=signupform.name.data,username=signupform.username.data, email=signupform.email.data, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template('login.html', signupform=signupform, loginform=loginform)

@app.route('/model', methods=["GET", "POST"])
def model():
    form = DiseaseField()  # Ensure this form has a file field named 'photo'
    if form.validate_on_submit():
        # Access the file from the form
        file = request.files['photo']  # Use 'photo' as the name of the file input
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_url = url_for('uploaded_file', filename=filename)
            print("Hello World")
            if file_url.startswith('/'):
                file_url = file_url[1:]
            print(file_url)
            # Return success response or redirect
            if session['language'] == "English":
                solution = disease_detection(file_url1=file_url,language="English")
                print("Hello Nakul")
                file_url = file_url[7:]
                return redirect(url_for('solution', solution=solution,file=file_url))
            else:
                file_url = file_url[7:]
                solution = disease_detection(file_url1=file_url,language="Gujarati")
                return redirect(url_for('solutionGuj', solution=solution,file=file_url))
    return render_template('model.html', form=form)


@app.route('/modelGuj', methods=["GET", "POST"])
def modelGuj():
    form = DiseaseField()  # Ensure this form has a file field named 'photo'
    session['language'] = "Gujarati"
    if form.validate_on_submit():
        # Access the file from the form
        file = request.files.get('photo')  # Use 'photo' as the name of the file input
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_url = url_for('uploaded_file', filename=filename)
            print("Hello World")
            if file_url.startswith('/'):
                file_url = file_url[1:]
            print(file_url)
            # Return success response or redirect
            if session['language'] == "English":
                solution = disease_detection(file_url1=file_url,language="English")
                file_url = file_url[7:]
                return redirect(url_for('solution', solution=solution,file=file_url))
            else:
                solution = disease_detection(file_url1=file_url,language="Gujarati")
                file_url = file_url[7:]
                return redirect(url_for('solutionGuj', solution=solution,file=file_url))
    return render_template('modelGuj.html', form=form)



def disease_detection(file_url1,language):
    print("hwkd")
    output = ImageProcessing(file_url1)
    solution = gemini(output,language)
    return solution

def gemini(disease,language):
    import os
    import google.generativeai as genai
    
    # Set API key as environment variable (replace with your actual path)
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\parwa\Downloads\farmbyte-main\farmbyte-main\client_secret_979856931806-kqceo61kebo7gr82k4f32g73q5rg3avv.apps.googleusercontent.com.json"

    # Configure the SDK
    genai.configure(api_key="AIzaSyD8yhwm7BDbqWDy4mohifzktpB18D9lXlU")


    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    )

    chat_session = model.start_chat(
    history=[
    ]
    )

    response = chat_session.send_message(f"Organic Solution for this disease {disease} in {language}")
    print(response.text)

    # print(response.text)
    return response.text

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/service')
def service():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('serviceGuj'))
    return render_template('service.html')


@app.route('/contact',methods=['GET','POST'])
def contact():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('contactGuj'))
    form = ContactForm()
    if form.validate_on_submit():
        new_contact = Contact(name=f"{form.name.data}",email=f"{form.email.data}",subject=form.subject.data,message=f"{form.message.data}")
        db.session.add(new_contact)
        db.session.commit()
        print("Info Added")
        print(form.name.data)
        print(form.email.data)
        print(form.subject.data)
        print(form.message.data)
        msg = Message(
        subject=f'Hello, Thanks for choosing FarmByte', 
        sender="nakuldesai2006@gmail.com",  # Ensure this matches MAIL_USERNAME
        recipients=[form.email.data]  # Replace with actual recipient's email
    )
        msg.body = f"Hello {form.name.data} we recieved your query for subject {form.subject.data} in {form.message.data}. We will look into it right away!!!."
        mail.send(msg)
        return redirect(url_for('home'))
    return render_template('contact.html',form=form)

@app.route('/policy')
def policy():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('policyGuj'))
    return render_template('policy.html')

@app.route('/solution/<solution>/<file>')
def solution(solution,file):
    return render_template('solution.html',solution=solution,file=file)

@app.route('/crop')
def crop():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('cropGuj'))
    return render_template('crop.html')

@app.route('/vendor')
def vendor():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('vendorGuj'))
    return render_template('vendor.html')

@app.route('/modern')
def modern():
    if session.get('language', 'English') == "Gujarati":
        return redirect(url_for('modernGuj'))
    return render_template('modern.html')


@app.route('/change_color', methods=["POST","GET"])
def change_color():
    data = request.json
    link_id = data.get('linkId', None)
    if link_id:
        session['active_link'] = link_id  # Store the clicked link ID in the session
        return jsonify({"success": True, "linkId": link_id})
    return jsonify({"success": True, "linkId": link_id}) 

@app.route('/get_active_link', methods=["GET"])
def get_active_link():
    link_id = session.get('active_link',False)  # Retrieve the stored link ID from the session
    return jsonify({"linkId": link_id})


# ----------------------------Gujarati------------------------
@app.route('/english')
def english():
    session['language'] = "English"
    return redirect(url_for('home'))


@app.route('/gujarati')
def homeGuj():
    session['language'] = "Gujarati"
    return render_template('indexGuj.html')

@app.route('/aboutGuj')
def aboutGuj():
    return render_template('aboutGuj.html')

@app.route('/exploreGuj')
def exploreGuj():
    return render_template('exploreGuj.html')

@app.route('/serviceGuj')
def serviceGuj():
    return render_template('serviceGuj.html')

@app.route('/contactGuj',methods=['GET','POST'])
def contactGuj():
    form = ContactForm()
    if form.validate_on_submit():
        new_contact = Contact(name=f"{form.name.data}",email=f"{form.email.data}",subject=form.subject.data,message=f"{form.message.data}")
        db.session.add(new_contact)
        db.session.commit()
        print("Info Added")
        msg = Message(
        subject=f'Hello, Thanks for choosing FarmByte', 
        sender="nakuldesai2006@gmail.com",  # Ensure this matches MAIL_USERNAME
        recipients=[form.email.data]  # Replace with actual recipient's email
    )
        msg.body = f"Hello {form.name.data},we recieved your query for subject {form.subject.data} in {form.message.data}. We will look into it right away!!!."
        mail.send(msg)
        return redirect(url_for('homeGuj'))
    return render_template('contactGuj.html',form=form)

@app.route('/policyGuj')
def policyGuj():
    return render_template('policyGuj.html')

@app.route("/solutionGuj<solution><file>")
def solutionGuj(solution,file):
    return render_template("solutionGuj.html",solution=solution,file=file)

@app.route('/cropGuj')
def cropGuj():
    return render_template('cropGuj.html')

@app.route('/vendorGuj')
def vendorGuj():
    return render_template('vendorGuj.html')

@app.route('/modernGuj')
def modernGuj():
    return render_template('modernGuj.html')

if __name__ == "__main__":
    app.run(debug=True)