from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, IntegerField, validators
from utils import generate_text
import keras

app = Flask(__name__)

new_model = keras.models.load_model('data/modelRNN_cuentosV2.h5')


class TextForm(Form):
    textx = TextAreaField('',
                          [validators.DataRequired(),
                           validators.length(min=5)])

    numberx = IntegerField('',
                            [validators.DataRequired()])


@app.route('/')
def index():
    form = TextForm(request.form)
    return render_template('text-form.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = TextForm(request.form)
    if request.method == 'POST' and form.validate():
        textx = request.form['textx']
        numberx = int(request.form['numberx'])
        text_rnn = generate_text(new_model, textx, numberx)
        return render_template('results.html',
                               content=textx,
                               text_rnn=text_rnn)
    return render_template('text-form.html', form=form)


if __name__ == '__main__':
    app.run()
