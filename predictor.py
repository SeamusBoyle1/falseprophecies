import lstm
from flask import Flask, jsonify
import json
import requests
from werkzeug.contrib.cache import SimpleCache

cache = SimpleCache()

app = Flask(__name__)


def denormalise_array(normalised, prenormalised, prenormalised_start):
    return [(float(prenormalised[prenormalised_start]) * (float(p) + 1)) for p in normalised]


@app.route('/')
def home():
    return '<h1>False Prophecies</h1>'


@app.route('/1/predict/<symbol>')
def hello_world(symbol):
    if symbol.lower() == "market":
        symbol = "%5Eaxjo"

    cache_key = 'predict_' + symbol
    rv = cache.get(cache_key)
    if rv:
        return rv

    epochs = 1
    seq_len = 50

    url_ = 'https://investor-api.herokuapp.com/api/1.0/analytics/shares/' + symbol + '/prices?range=2y&interval=1d'

    try:
        r = requests.get(url_)
        orig_data = json.loads(r.text)
    except (requests.exceptions.RequestException, ValueError):
        return jsonify({})

    clean_data = list(filter(None, orig_data))

    X_train, y_train, X_test, y_test = lstm.load_data2(clean_data, seq_len, True)

    model = lstm.build_model([1, 50, 100, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    # predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    predicted = lstm.predict_point_by_point(model, X_test)

    denormalised_prediction = denormalise_array(predicted, clean_data, len(clean_data) - len(y_test))

    json_ = jsonify({'prediction': denormalised_prediction,
                     'true_data': clean_data[-len(denormalised_prediction):]})

    cache.set(cache_key, json_, timeout=8 * 60 * 60)

    return json_
