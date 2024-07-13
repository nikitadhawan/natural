import os 
os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import joblib

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

def save_path_fn(cfg, final=False):
    save_path = cfg.save_path + cfg.data.dataset + "/"
    if final:
        save_path += cfg.date_time + "_"
    save_path += cfg.data.dataset + "_" + cfg.data.split + "_" + cfg.experiment_name 
    save_path += "_" + cfg.baseline_name + "_preds.csv"
    return save_path 

def get_encodings(posts, baseline_name, split, save_path):
    if baseline_name == "sentence_encoder":
        encoder = SentenceTransformer("all-mpnet-base-v2")
        encodings = np.array([encoder.encode(x) for x in posts])
    elif baseline_name == "bow":
        encodings = np.array([x for x in posts])
        if split=="train":
            vectorizer = CountVectorizer() 
            encodings = vectorizer.fit_transform(encodings)
            joblib.dump(vectorizer, save_path.replace("_preds.csv", "_vectorizer.pkl"))
        else:
            vectorizer = joblib.load(save_path.replace("_preds.csv", "_vectorizer.pkl").replace("val", "train"))
            encodings = vectorizer.transform(encodings)
    return encodings

def get_predictor(X, Y):
    predictor = MLPClassifier(tol=1e-4, max_iter=20, early_stopping=True) 
    hidden_grid = [128, 256, 512]
    n_layers_grid = [2, 3]
    parameter_grid = {
        'hidden_layer_sizes': [[hidden_dim for _ in range(n_layers)] for hidden_dim in hidden_grid for n_layers in n_layers_grid],
        'activation': ['tanh', 'relu'],
        'learning_rate_init': [1e-4, 5e-3, 1e-3],
        'batch_size': [32, 64],
        'solver': ['sgd', 'adam']
    }
    predictor = GridSearchCV(predictor, parameter_grid, scoring='accuracy', n_jobs=16, refit=True)

    predictor = predictor.fit(X, Y)
    return predictor

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    dataset = instantiate(cfg.data.dataset_class)
    X_cols, Y_cols, Z_cols, strat_cols = dataset.XYZ_names(return_strat_cols=True)
    llm_posts_df = pd.read_csv(cfg.data.posts_path).head(cfg.data.dataset_size)
    save_path = save_path_fn(cfg)
    encodings = get_encodings(llm_posts_df["llm_post"], cfg.baseline_name, cfg.data.split, save_path)
    X, Z, Y = dataset.get_data_split(transform=True, obs=True)[cfg.data.split]
    ground_truth_data = X.copy()[strat_cols]
    ground_truth_data[Y_cols] = Y 
    ground_truth_data[Z_cols] = Z

    pred_features = pd.DataFrame()
    for feature in strat_cols + [Y_cols, Z_cols]:

        targets = ground_truth_data[[feature]].copy()
        num_classes = targets[feature].nunique()
        targets = pd.get_dummies(targets, columns=[feature,]).to_numpy().reshape((encodings.shape[0], num_classes))

        if cfg.data.split == "train":
            predictor = get_predictor(encodings, targets)
            joblib.dump(predictor, save_path.replace("_preds.csv", "_" + feature + "_classifier.pkl"))
        else:
            model_path = save_path.replace("_preds.csv", "_" + feature + "_classifier.pkl")
            model_path = model_path.replace("val", "train")
            predictor = joblib.load(model_path)

        acc = predictor.score(encodings, targets)

        all_preds = predictor.predict(encodings)
        all_preds = np.argmax(all_preds, axis=1)
        pred_features.loc[:len(all_preds)-1, feature] = all_preds
        pred_features.loc[cfg.data.split + "_acc", feature] = acc
        pred_features.to_csv(save_path)

    pred_features.to_csv(save_path)


if __name__ == "__main__":
    main()