from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField,FileField,SelectField,TextAreaField
from wtforms.validators import DataRequired, EqualTo

class LoginForm(FlaskForm):
    username = StringField('username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    name = StringField('Username', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class DiseaseField(FlaskForm):
    photo = FileField('Uplaod Your Photo', validators=[DataRequired()])
    submit = SubmitField('Upload')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    subject = SelectField("Subject", choices=[
        ("Subject", "Subject"),
        ("Technical Error", "Technical Error"),
        ("Difficulty understanding Information", "Difficulty understanding Information"),
        ("Problem related to Disease Identification", "Problem related to Disease Identification"),
        ("Other crop related problems","Other crop related problems ")
    ])
    message = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField("Send Message")

class LangForm(FlaskForm):
    choose = SelectField("Subject", choices=[
        ("Choose Language", "Choose Language"),
        ("English", "English"),
        ("Gujarati", "Gujarati")
        
    ])

