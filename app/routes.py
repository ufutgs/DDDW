from app import application
from flask import render_template, flash
from app.forms import  CreateQuestionForm, ChallengeAnswerForm
import app.serverlibrary as lit

@application.route('/')
@application.route('/thisistask1')
def index():
	beta , maxi,mini= lit.start()
	print(maxi,mini)
	return render_template('index.html', title='Home',beta=beta,maxi=maxi,mini=mini)




