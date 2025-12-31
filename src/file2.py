import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import dagshub
dagshub.init(repo_owner="Richa300587", repo_name="Experiment-Tracking-with-ML-Flow", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Richa300587/Experiment-Tracking-with-ML-Flow.mlflow")
mlflow.set_experiment("Wine_Classification_Experiment")
wine_data = load_wine()
X = wine_data.data
y = wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=12, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)   

    mlflow.set_tag("Author", "Richa")
    mlflow.set_tag("Project", "Wine_Classification")
    mlflow.log_param("n_estimators", 12)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", 6)

    #creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf, "RandomForestWineModel")

    print(f"Accuracy: {accuracy}")