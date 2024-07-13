from hydra.utils import instantiate

from causality_llm.estimation.cre import CRE
from causality_llm.models import CAUSAL_MODELS

class OBS(CRE):

    def __init__(self, dataset, name="obs"):
        self.name = name
        self.dataset = dataset
        self.observational = True
        self.eval_propensity = False
        self.eval_imputation = False
        self.eval_accuracy = False

    def get_ate(self, cfg, indices=None, model_name="ipw"):
        X_cols, _, _ = self.dataset.XYZ_names()
        data_dict = self.dataset.get_data_split(obs=cfg.data.observational)
        X_train, Z_train, Y_train = data_dict["train"]
        X_val, Z_val, Y_val = data_dict["val"]
        X_train, X_val = X_train[X_cols], X_val[X_cols]
        if indices is not None:
            X_train, Z_train, Y_train = X_train.iloc[indices], Z_train.iloc[indices], Y_train.iloc[indices]

        data = (X_train, X_val, Z_train.squeeze(), Z_val.squeeze(), Y_train.squeeze(), Y_val.squeeze())
        effect_dict = {}
        model = CAUSAL_MODELS[model_name]()
        train_effect, val_effect = model.get_effect(data)
        effect_dict["train_" + model_name] = train_effect
        effect_dict["val_" + model_name] = val_effect

        return effect_dict
