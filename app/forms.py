from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectMultipleField, IntegerField, HiddenField
from wtforms.validators import DataRequired, ValidationError, EqualTo


class CreateQuestionForm(FlaskForm):
	expression = StringField('Math Expression', validators=[DataRequired()])
	assign_to = SelectMultipleField('Send To', validators=[DataRequired()])
	submit = SubmitField('Submit')


class ChallengeAnswerForm(FlaskForm):
	challenge_id = HiddenField('Challenge ID')
	answer = StringField('Answer', validators=[DataRequired()])
	elapsed_time = HiddenField('Elapsed Time')
	submit = SubmitField('Submit')




	