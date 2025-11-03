from zenml import step
from sklearn.metrics import classification_report
import mlflow
import logging

logging.basicConfig(level=logging.INFO)

@step
def evaluation(X_test, y_test,model,run_id):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    with mlflow.start_run(run_id=run_id):

        mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'f1_score': report['macro avg']['f1-score'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall']
        })
        
    return report