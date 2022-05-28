import sqlite3

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

# Librerias para importar el modelo
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)


# -------Operaciones DML en BDsqlite
def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO sentimientos_db (texto, sentimento, fecha)" \
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


def sqlite_select(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT texto, sentimento, fecha FROM sentimientos_db")
    results = c.fetchall()
    return results


db = os.path.join(cur_dir, 'DB_sentimientos_Esp.sqlite')
# ------- Cargando el modelo de IA (regresión logistica)

modeloPLNRegLog = pickle.load(
    open(os.path.join(cur_dir, 'modeloIA', 'LogRegression_PLN_classSentimientos_model.sav'), 'rb'))
vocabulary = pickle.load(open(os.path.join(cur_dir, 'modeloIA', 'vocabulary.pkl'), 'rb'))
vectorizer = CountVectorizer(min_df=0, lowercase=True,
                             vocabulary=vocabulary)  # creando un nuevo vectorizador con el vocabulario cargado


# Funcion que usaremos para clasificar el texto del usuario
def f_clasificar(texto):
    label = {0: 'Negativo', 1: 'Positivo'}
    oracion = [texto]
    x_bag = vectorizer.transform(oracion)
    predict = modeloPLNRegLog.predict(x_bag)[0]
    return label[predict]


def f_revertir(prediccion):
    if prediccion == "Positivo":
        return "Negativo"
    else:
        return "Positivo"


def f_enviar_imagen(prediccion):
    if prediccion == "Positivo":
        return "https://definicion.de/wp-content/uploads/2013/06/signo-mas.png"
    else:
        return "https://us.123rf.com/450wm/djdarkflower/djdarkflower1603/djdarkflower160300001/53823525-abajo-pulgares-icono-pulgar-rojo-muestra-un-bot%C3%B3n-mano-s%C3%ADmbolo-tela-no-negativo-malo-aversi%C3%B3n-intern.jpg?ver=6"


class evaluarForm(Form):
    evaluarText = TextAreaField('', [validators.DataRequired(),
                                     validators.length(min=15)])


@app.route('/')
def index():
    return render_template('Inicio.html')


@app.route('/empezar')
def empezar():
    form = evaluarForm(request.form)
    return render_template('index.html', form=form)


@app.route('/resultadoIA', methods=['POST'])
def resultadoIA():
    form = evaluarForm(request.form)
    if request.method == 'POST' and form.validate():
        texto = request.form['evaluarText']
        predict = f_clasificar(texto)
        pred_contra = f_revertir(predict)
        imagen_contra = f_enviar_imagen(pred_contra)
        imagen_actu = f_enviar_imagen(predict)
        return render_template('resultado.html', content=texto, prediction=predict, pred_contra=pred_contra,
                               imagen_contra=imagen_contra, imagen_actu=imagen_actu)
    return render_template('index.html', form=form)


@app.route('/gracias', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    texto = request.form['texto']
    prediction = request.form['prediction']
    inv_label = {'Positivo': 1, 'Negativo': 0}
    y = inv_label[prediction]
    sqlite_entry(db, texto, y)
    return render_template('gracias.html')


@app.route('/ReportUser', methods=['POST'])
def sqliteReport():
    dataset = sqlite_select(db)
    return render_template('reportUser.html', dataset=dataset)


if __name__ == '__main__':
    app.run(debug=True)
