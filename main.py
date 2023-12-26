from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('best_model2.pkl', 'rb') as f:
    model2 = pickle.load(f)
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get parameters from the POST request
    body = request.get_json()
    print(body)

    # Convert parameters into numpy array
    v1 = body['v1']
    v2 = body['v2']
    v3 = body['v3']
    v4 = body['v4']
    v5 = body['v5']
    v6 = body['v6']
    v7 = body['v7']
    v8 = body['v8']
    v9 = body['v9']
    v10 = body['v10']
    v11 = body['v11']
    v12 = body['v12']
    v13 = body['v13']
    v14 = body['v14']
    v15 = body['v15']
    v16 = body['v16']
    v17 = body['v17']
    v18 = body['v18']
    v19 = body['v19']
    v20 = body['v20']
    v21 = body['v21']
    v22 = body['v22']
    v23 = body['v23']
    v24 = body['v24']
    v25 = body['v25']
    v26 = body['v26']
    v27 = body['v27']
    v28 = body['v28']
    amount = body['amount']

    # make numpy array from params
    np_array = [
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
        v16,
        v17,
        v18,
        v19,
        v20,
        v21,
        v22,
        v23,
        v24,
        v25,
        v26,
        v27,
        v28,
        # time,
        amount
    ]

    # Make prediction using the loaded model
    prediction = model.predict(np.array(np_array).reshape(1, -1))

    # Convert numpy array to list and return as JSON
    print(prediction)
    output = prediction[0].tolist()
    return jsonify(output)


@app.route('/predict2', methods=['POST'])
def predict2():
    # Get parameters from the POST request
    body = request.get_json()
    print(body)

    # Convert parameters into numpy array
    distance_from_home = body['distanceFromHome']
    distance_from_last_transaction = body['distanceFromLastTransaction']
    ratio_to_median_purchase_price = body['ratioToMedianPurchasePrice']
    repeat_retailer = body['repeatRetailer']
    used_chip = body['usedChip']
    used_pin_number = body['usedPinNumber']
    online_order = body['onlineOrder']

    # make numpy array from params
    np_array = [
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        repeat_retailer,
        used_chip,
        used_pin_number,
        online_order,
    ]

    # Make prediction using the loaded model
    x = np.array(np_array).reshape(1, -1)
    print(x)
    prediction = model2.predict(x)

    # Convert numpy array to list and return as JSON
    print(prediction)
    output = prediction[0].tolist()
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)


@app.route('/predictTest', methods=['POST'])
def predict_test():
    # Get parameters from the POST request
    body = request.get_json()

    np_array = np.array(body['features'])

    # Make prediction using the loaded model
    prediction = model.predict(np_array.reshape(1, -1))

    # Convert numpy array to list and return as JSON
    print(prediction)
    output = prediction[0].tolist()
    return jsonify(output)

