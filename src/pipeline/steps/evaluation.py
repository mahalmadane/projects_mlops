from zenml import step
from sklearn.metrics import accuracy_score
from typing import NamedTuple

class Accuracy_and_BestScore(NamedTuple):
    accuracy: dict
    best_score: str

@step
def evaluation(model1, model2, X_test, y_test):
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy2 = accuracy_score(y_test, y_pred2)
    accuracy = {
        "Logistic": accuracy1,
        "K-Neighbors": accuracy2
    }
    best_score = max(accuracy, key=accuracy.get)

    return Accuracy_and_BestScore(accuracy, best_score)