from sklift.datasets import fetch_x5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class RetailHero(object):

    def __init__(self, feature_name="age_binary", prop_score=0.3, train_size=2000, seed=10):
        self.ground_truth = 0.033217
        self.uncorrected_ground_truth = 0.015295
        self.feature_name = feature_name # confounding feature 
        self.prop_score = prop_score # propensity score
        self.train_size = train_size # size of training set
        self.seed = seed
        self.num_outcomes = 2
        self.transform_maps = {"avg_purchase": {0: "1 - 263", 1: "264 - 396", 2: "397 - 611", 3: "612 - 7694"}, 
                               "avg_product_quantity": {0: "No", 1: "Yes"}, 
                               "num_transactions": {0: "1 - 8", 1: "9 - 15", 2: "16 - 27", 3: "28 - 320"},
                               "avg_regular_points_received": {0: "No", 1: "Yes"},
                               "sample_avg_purchase": {0: "1 - 263", 1: "264 - 396", 2: "397 - 611", 3: "612 - 7694"}, 
                               "sample_avg_product_quantity": {0: "No", 1: "Yes"}, 
                               "sample_num_transactions": {0: "1 - 8", 1: "9 - 15", 2: "16 - 27", 3: "28 - 320"},
                               "sample_avg_regular_points_received": {0: "No", 1: "Yes"},}

    def XYZ_names(self, return_enum_cols=False, return_strat_cols=False):
        X_cols = ["avg_purchase", "avg_product_quantity", "num_transactions", "age", "avg_regular_points_received"]
        Y_cols = "purchase"
        Z_cols = "sms_communication"
        if return_enum_cols:
            # these are the features to be enumerated if we are enumerating X
            return X_cols, Y_cols, Z_cols, ["avg_regular_points_received", "age_binary"]
        if return_strat_cols:
            return X_cols, Y_cols, Z_cols, ["age_binary", "avg_purchase", "avg_product_quantity", "num_transactions", "avg_regular_points_received"]
        return X_cols, Y_cols, Z_cols
    
    def get_data_split(self, transform=True, obs=False):
        try:
            data_path = "/private/home/nikitadhawan/causality_llm/data/retail_hero/retail_hero_" 
            if obs:
                data_path += "obs_"
            x_path = data_path 
            if transform:
                x_path += "transformed_"
            data_path += "prop" + str(self.prop_score) + "_" 
            x_path += "prop" + str(self.prop_score) + "_" 
            X_train = pd.read_csv(x_path + "train" + str(self.train_size) + "_x.csv", index_col=0)
            Y_train = pd.read_csv(data_path + "train" + str(self.train_size) + "_y.csv", index_col=0)
            Z_train = pd.read_csv(data_path + "train" + str(self.train_size) + "_z.csv", index_col=0)
            X_val = pd.read_csv(x_path + "val" + str(self.train_size) + "_x.csv", index_col=0)
            Y_val = pd.read_csv(data_path + "val" + str(self.train_size) + "_y.csv", index_col=0)
            Z_val = pd.read_csv(data_path + "val" + str(self.train_size) + "_z.csv", index_col=0)
        except:
            dataset = self.get_retail_hero_data(transform=transform)
            if obs:
                dataset = self.make_obs(dataset, feature_name=self.feature_name, 
                                        feature_propensities=[self.prop_score, 1-self.prop_score], 
                                        seed=self.seed)
            X_train, X_val, Z_train, Z_val, Y_train, Y_val = self.get_retail_hero_split(dataset, self.seed, train_size=self.train_size)
        return {"train": (X_train, Z_train, Y_train), "val": (X_val, Z_val, Y_val)}

    def make_obs(self, dataset, feature_name, feature_propensities, seed):
        np.random.seed(seed)
        X = dataset.features[feature_name]
        ex = feature_propensities[0] * (1 - X) + feature_propensities[1] * X
        n = len(dataset.features)
        n1 = np.sum(dataset.treatment.values)
        n0 = n - n1
        # multiply propensity scores with empirical p_X 
        p_xyt_obs = dataset.treatment * ex * (1/n1) + (1 - dataset.treatment) * (1 - ex) * (1/n0) 
        p_xyt_obs /= np.sum(p_xyt_obs) 
        sampled_indices = np.random.choice(range(n), p=p_xyt_obs, size=n)
        dataset.features = dataset.features.iloc[sampled_indices]
        dataset.treatment = dataset.treatment.iloc[sampled_indices]
        dataset.target = dataset.target.iloc[sampled_indices]
        return dataset
    
    def age_transform(self, dataset):
        dataset.features.loc[dataset.features["age"] <= 45, "age"] = 0
        dataset.features.loc[dataset.features["age"] > 45, "age"] = 1
        return dataset.features.age

    def transform_cat(self, dataset, feat_name):
        bin_seq = {"avg_purchase": [0, 263, 396, 611, 7694.1], 
                "avg_product_quantity": [-0.1, 7, 120.1], 
                "num_transactions": [0, 8, 15, 27, 320.1], 
                "avg_regular_points_received": [-0.1, 5, 205]}
        transformed = pd.cut(dataset.features[feat_name], bins=bin_seq[feat_name], labels=np.arange(len(bin_seq[feat_name]) - 1))
        dataset.features[feat_name] = transformed.astype("int8")
        return dataset.features[feat_name]
        
    def get_retail_hero_data(self, transform=True):
        # https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
        dataset = fetch_x5()
        df_features = dataset.data["purchases"].set_index("client_id")
        num_transactions = df_features.groupby("client_id").agg(num_transactions=pd.NamedAgg(column="transaction_id", aggfunc="nunique"))
        df_features = df_features.drop(["transaction_datetime", "store_id", "product_id", "trn_sum_from_red", "trn_sum_from_iss", 
                                        "express_points_received", "regular_points_spent", "express_points_spent"], axis=1)
        df_features_grouped = df_features.groupby(["client_id","transaction_id"]).agg(
            purchase_sum=pd.NamedAgg(column="purchase_sum", aggfunc="last"),
            regular_points_received=pd.NamedAgg(column="regular_points_received", aggfunc="last"),
            product_quantity=pd.NamedAgg(column="product_quantity", aggfunc="sum")
        )
        df_features = df_features_grouped.groupby("client_id").agg(
            avg_purchase=pd.NamedAgg(column="purchase_sum", aggfunc="mean"),
            avg_regular_points_received=pd.NamedAgg(column="regular_points_received", aggfunc="mean"),
            avg_product_quantity=pd.NamedAgg(column="product_quantity", aggfunc="mean")
        )   
        df_features = df_features.merge(num_transactions, left_index=True, right_index=True)
        age = dataset.data["clients"].set_index("client_id")["age"]
        df_features = df_features.merge(age, left_index=True, right_index=True)
        df_train = pd.concat([dataset.data["train"], dataset.treatment , dataset.target], axis=1).set_index("client_id")
        dataset.features = df_features.loc[df_train.index, :]
        dataset.features.loc[dataset.features["age"] <= 45, "age_binary"] = 0
        dataset.features.loc[dataset.features["age"] > 45, "age_binary"] = 1
        dataset.treatment = df_train["treatment_flg"].astype("int8")
        dataset.target = df_train["target"].astype("int8")

        if transform:
            self.age_transform(dataset)
            for name in ["avg_purchase", "avg_product_quantity", "num_transactions", "avg_regular_points_received"]:
                self.transform_cat(dataset, name)
        return dataset

    def get_retail_hero_split(self, dataset, seed, test_size=2000, train_size=500):
        stratify_cols = pd.concat([dataset.treatment, dataset.target], axis=1)
        X_train, X_val, Z_train, Z_val, Y_train, Y_val = train_test_split(
            dataset.features, # 4 features: "avg_purchase", "avg_product_quantity", "num_transactions", "age", "avg_regular_points_received"
            dataset.treatment,
            dataset.target,
            stratify=stratify_cols,
            train_size=train_size,
            test_size=test_size,
            random_state=seed
        )
        return X_train, X_val, Z_train, Z_val, Y_train, Y_val
    
    def get_options(self, feature_names):
        options = {"avg_purchase": ["1 - 263", "264 - 396", "397 - 611", "612 - 7694"], 
                   "avg_product_quantity": ["No", "Yes"], 
                   "avg_regular_points_received": ["No", "Yes"], 
                   "num_transactions": ["1 - 8", "9 - 15", "16 - 27", "28 - 320"],
                   "age_binary": ["No", "Yes"],
                   "sms_communication": ["No", "Yes"], # treatment
                   "purchase": ["No", "Yes"] # outcome
                } 
        return {key: options[key] for key in feature_names}


    def get_question_prompt(self, feature_names):
        questions = {
            "avg_purchase": "In what interval does the user's average purchase per transaction lie?",
            "avg_product_quantity": "Did the user purchase more than 7 products on average?",
            "avg_regular_points_received": "Has the user received more than 5 regular points on average?",
            "num_transactions": "In what interval does the user's number of transactions so far lie?",
            "age_binary": "Is the user's age greater than 45 years?", 
            "sms_communication": "Did the user mention receiving marketing communication via SMS from the retailer?", # treatment
            "purchase": "Did the user mention a recent purchase from the retailer?" # outcome
        } 
        return {key: questions[key] for key in feature_names}

    def post_gen_specs(self, feat_dict):
        descriptions = {
            "avg_purchase": "The average dollar value you have spent per transaction: ",
            "avg_product_quantity": "The rough number of products you tend to purchase in each transaction: ",
            "avg_regular_points_received": "Average number of regular points you have received on a transaction: ",
            "num_transactions": "Total number of transactions you have made so far: ",
            "age": "Mention your age, in years, which is: ", 
            "sms_communication": "You received marketing communication via SMS from the retailer - true or false: ", # treatment
            "purchase": "You made another purchase from the retailer recently - true or false: " # outcome
        } 
        interpreted = {key: feat_dict[key] for key in ["sms_communication", "purchase"] if key in feat_dict.keys()}
        interpreted = self.interpret_samples(interpreted)
        feat_dict.update(interpreted)
        rounded = {key: round(feat_dict[key]) for key in ["avg_purchase", "avg_product_quantity", "avg_regular_points_received", "num_transactions", "age"] if key in feat_dict.keys()}
        feat_dict.update(rounded)
        description_text = ""
        for feat in feat_dict.keys():
            description_text += descriptions[feat] + str(feat_dict[feat]) + ". \n "
        return description_text

    def transform_samples(self, dct):
        all_keys = list(dct.keys())
        for field in all_keys:
            if field in ["sample_sms_communication", "sample_purchase", "sms_communication", "purchase"]:
                dct[field] = 0 if dct[field] == "No" else 1
            elif field in ["sample_age_binary", "age_binary"]:
                age_map = {"No": 0, "Yes": 1}
                dct[field] = age_map[dct[field]]
            elif field in list(self.transform_maps.keys()):
                map = self.transform_maps[field]
                invert_map = {v: k for k, v in map.items()}
                dct[field] = invert_map[dct[field]]
        return dct

    def interpret_samples(self, dct):
        all_keys = list(dct.keys())
        for field in all_keys:
            if field in ["sample_sms_communication", "sample_purchase", "sms_communication", "purchase"]:
                dct[field] = "No" if dct[field] == 0 else "Yes"
            elif field in ["sample_age_binary", "age_binary"]:
                age_map = {0: "No", 1: "Yes"}
                dct[field] = age_map[dct[field]]
            elif field in list(self.transform_maps.keys()):
                map = self.transform_maps[field]
                dct[field] = map[dct[field]]
        return dct
    
    def discretize(self, sample_df, hard_filter=True):
        sample_df = sample_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
        if "age" in sample_df.columns:
            sample_df["age"] = pd.to_numeric(sample_df["age"], errors='coerce')
            sample_df.loc[sample_df["age"] <= 45, "age_binary"] = 0
            sample_df.loc[sample_df["age"] > 45, "age_binary"] = 1
            age_unique = [i for i in sample_df["age"].unique() if not np.isnan(i)]
            sample_df["age"] = sample_df["age"].map(lambda x: x if not np.isnan(x) else np.random.choice(age_unique))
            sample_df["age_binary"] = sample_df["age_binary"].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1]))

        bin_seq = {"avg_purchase": [0, 263, 396, 611, 7694.1], 
                "avg_product_quantity": [-0.1, 7, 120.1], 
                "num_transactions": [0, 8, 15, 27, 320.1], 
                "avg_regular_points_received": [-0.1, 5, 205]}
        for feat_name in bin_seq.keys():
            if feat_name in sample_df.columns:
                sample_df[feat_name] = sample_df[feat_name].apply(lambda x: float(x))
                transformed = pd.cut(sample_df[feat_name], bins=bin_seq[feat_name], labels=np.arange(len(bin_seq[feat_name]) - 1))
                sample_df[feat_name] = transformed
                sample_df[feat_name] = sample_df[feat_name].map(lambda x: x if not np.isnan(x) else np.random.choice(np.arange(len(bin_seq[feat_name]) - 1)))
        
        for field in ["purchase", "sms_communication"]:
            if field in sample_df.columns:
                field_map = {"No": 0, "Yes": 1}
                sample_df[field] = sample_df[field].replace(field_map)
                sample_df[field] = sample_df[field].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1]))

        return sample_df