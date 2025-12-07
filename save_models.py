import mlflow
import joblib
import os

os.makedirs('models', exist_ok=True)

#i save models in local directory for docker
models = {
    'vectorizer': 'runs:/1f71bd4dffca4a7696a1fdb6ca6ae9d0/tfidfvectorizer',
    'oneclass_svm': 'runs:/03f42d590f3c46f094330623dace83fc/oneclasssvm', 
    'rfc': 'runs:/848088676de540cabae2b392671bc6e1/rfc'
}

for name, uri in models.items():
    print(f"Скачиваю {name}...")
    model = mlflow.sklearn.load_model(uri)
    joblib.dump(model, f'models/{name}.joblib', compress=3)

print("✅ Модели сохранены в папке models/")