from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import joblib

class RecencyModelTraining:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        rmse = root_mean_squared_error(y, predictions, squared=False)
        return rmse

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)
