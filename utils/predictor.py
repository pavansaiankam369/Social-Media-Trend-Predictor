import numpy as np

def predict_trends(model, data, scaler, look_back, future_days=3):
    last_sequence = data[-look_back:]
    predictions = []
    current_sequence = last_sequence.reshape((1, look_back, 1))
    for _ in range(future_days):
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions