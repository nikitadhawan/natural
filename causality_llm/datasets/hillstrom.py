from sklift.datasets import fetch_hillstrom
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Hillstrom(object):

    def __init__(self, feature_name="newbie", prop_score=0.3, train_size=2000, seed=10):
        self.ground_truth = 0.060865
        self.uncorrected_ground_truth = 0.023771
        self.feature_name = feature_name # confounding feature 
        self.prop_score = prop_score # propensity score
        self.train_size = train_size # size of training set
        self.seed = seed
        self.num_outcomes = 2

    def XYZ_names(self, return_enum_cols=False, return_strat_cols=False):
        X_cols = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
        Y_cols = "visit"
        Z_cols = "email_type"
        if return_enum_cols:
            # these are the features to be enumerated if we are enumerating X
            return X_cols, Y_cols, Z_cols, ["womens", "newbie"]
        if return_strat_cols:
            return X_cols, Y_cols, Z_cols, ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
        return X_cols, Y_cols, Z_cols
    
    def get_data_split(self, transform=True, obs=False):
        try:
            data_path = "/private/home/nikitadhawan/causality_llm/data/hillstrom/hillstrom_" 
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
            dataset = self.get_hillstrom_data(transform=transform)
            if obs:
                dataset = self.make_obs(dataset, feature_name=self.feature_name, 
                                        feature_propensities=[self.prop_score, 1-self.prop_score], 
                                        seed=self.seed)
            X_train, X_val, Z_train, Z_val, Y_train, Y_val = self.get_hillstrom_split(dataset, self.seed, train_size=self.train_size)
        return {"train": (X_train, Z_train, Y_train), "val": (X_val, Z_val, Y_val)}

    def make_obs(self, dataset, feature_name, feature_propensities, seed):
        np.random.seed(seed)
        X = dataset.data[feature_name]
        ex = feature_propensities[0] * (1 - X) + feature_propensities[1] * X
        n = len(dataset.data)
        n1 = np.sum(dataset.treatment_binary.values)
        n0 = n - n1
        # multiply propensity scores with empirical p_X 
        p_xyt_obs = dataset.treatment_binary * ex * (1/n1) + (1 - dataset.treatment_binary) * (1 - ex) * (1/n0)
        p_xyt_obs /= np.sum(p_xyt_obs) 
        sampled_indices = np.random.choice(range(n), p=p_xyt_obs, size=n)
        dataset.data = dataset.data.iloc[sampled_indices]
        dataset.treatment_binary = dataset.treatment_binary.iloc[sampled_indices]
        dataset.target = dataset.target.iloc[sampled_indices]
        return dataset

    def hillstrom_historic_segment_transform(self, dataset):
        for payment in dataset.data['history_segment'].unique(): 
            if payment =='1) $0 - $100':
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 0 #50
            elif payment =='2) $100 - $200':
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 1 #150
            elif payment =='3) $200 - $350':
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 2 #275
            elif payment =='4) $350 - $500':
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 3 #425
            elif payment =='5) $500 - $750':
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 4 #575
            elif payment =='6) $750 - $1000':
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 5 #825
            else:
                dataset.data.loc[dataset.data['history_segment'] == payment, 'history_segment'] = 6 #1000
        return dataset.data.history_segment

    def hillstrom_zipcode_transform(self, dataset):
        for zipcode_type in dataset.data['zip_code'].unique(): 
            if zipcode_type =='Suburban':
                dataset.data.loc[dataset.data['zip_code'] == zipcode_type, 'zip_code'] = 0
            elif zipcode_type =='Rural':
                dataset.data.loc[dataset.data['zip_code'] == zipcode_type, 'zip_code'] = 1
            else:
                dataset.data.loc[dataset.data['zip_code'] == zipcode_type, 'zip_code'] = 2
        return dataset.data.zip_code

    def hillstrom_channel_transform(self, dataset):
        for channel_type in dataset.data['channel'].unique(): 
            if channel_type =='Phone':
                dataset.data.loc[dataset.data['channel'] == channel_type, 'channel'] = 0
            elif channel_type =='Web':
                dataset.data.loc[dataset.data['channel'] == channel_type, 'channel'] = 1
            else:
                dataset.data.loc[dataset.data['channel'] == channel_type, 'channel'] = 2
        return dataset.data.zip_code

    def hillstrom_treatment_transform(self, dataset):
        treat_dict = {
                    'Womens E-Mail': 0,
                    'Mens E-Mail': 1,
                    'No E-Mail': 2 
                    }
        dataset.treatment = dataset.treatment.map(treat_dict)
        dataset.treatment = dataset.treatment.astype('int8')
        return dataset.treatment
        
    def get_hillstrom_data(self, binary_treat=True, transform=True):
        # https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
        dataset = fetch_hillstrom()
        if binary_treat:
            treat_dict_binary = {
                        'Womens E-Mail': 1,
                        'No E-Mail': 0, 
                        'Mens E-Mail': 1
                        }
            dataset.treatment_binary = dataset.treatment.map(treat_dict_binary)
            dataset.treatment_binary = dataset.treatment_binary.astype('int8')
        if transform:
            self.hillstrom_historic_segment_transform(dataset)
            self.hillstrom_zipcode_transform(dataset)
            self.hillstrom_channel_transform(dataset)
            self.hillstrom_treatment_transform(dataset)
        dataset.target = dataset.target.astype('int8')
        return dataset

    def get_hillstrom_split(self, dataset, seed, test_size=2000, train_size=500):
        stratify_cols = pd.concat([dataset.treatment_binary, dataset.target], axis=1)
        X_train, X_val, Z_train, Z_val, Y_train, Y_val = train_test_split(
            dataset.data, # 8 features: 'recency', 'history_segment', 'history', 'mens', 'womens', 'zip_code', 'newbie', 'channel'
            dataset.treatment_binary,
            dataset.target,
            stratify=stratify_cols,
            train_size=train_size,
            test_size=test_size,
            random_state=seed
        )
        return X_train, X_val, Z_train, Z_val, Y_train, Y_val
    
    def get_options(self, feature_names):
        options = {
            "recency": ["1 - 4", "5 - 8", "9 - 12"],
            "history": ["0 - 100", "100 - 200", "200 - 350", "350 - 500", "500 - 750", "750 - 1000", "1000 +"],
            "mens": ["No", "Yes"],
            "womens": ["No", "Yes"],
            "zip_code": ["Suburban area", "Rural area", "Urban area"],
            "newbie": ["No", "Yes"],
            "channel": ["Phone", "Web", "Multichannel"],
            "email_type": ["No", "Yes"], # treatment
            "visit": ["No", "Yes"] # outcome
        } 
        return {key: options[key] for key in feature_names}


    def get_question_prompt(self, feature_names):
        questions = {
            "recency": "How many months has it been since the user's last purchase?",
            "history": "What range does the dollar value spent by the user previously lie in?",
            "mens": "Did the user purchase men's merchandise?",
            "womens": "Did the user purchase women's merchandise?",
            "zip_code": "Which area does the user's zipcode lie in?",
            "newbie": "Was the user was a newbie, or a new customer, in the past one year?",
            "channel": "What channel did the user use to make their previous purchase?",
            "email_type": "Did the user mention receiving a marketing email?", # treatment
            "visit": "Did the user visit the website recently?" # outcome
        } 
        return {key: questions[key] for key in feature_names}
    
    def post_gen_specs(self, feat_dict):
        descriptions = {
            "recency": "The number of months since your last purchase: ",
            "history": "The dollar value you spent in the past one year is: ",
            "mens": "Your previous purchase was of men's merchandise: ",
            "womens": "Your previous purchase was of women's merchandise: ",
            "zip_code": "Whether you live in an Urban, Suburban or Rural area: ",
            "newbie": "You became a new customer in the past one year: ",
            "channel": "Whether you made your purchase in the past one year using a Phone, the Web or Multiple channels: ",
            "email_type": "You received a marketing e-mail about women's or men's merchandise: ", # treatment is only going to be true if mentioned
            "visit": "You visited the website again recently: " # outcome
        } 
        interpreted = {key: feat_dict[key] for key in ["mens", "womens", "newbie", "email_type", "visit"] if key in feat_dict.keys()}
        interpreted = self.interpret_samples(interpreted)
        feat_dict.update(interpreted)
        description_text = ""
        for feat in feat_dict.keys():
            description_text += descriptions[feat] + str(feat_dict[feat]) + ". \n"
        return description_text

    def transform_samples(self, dct):
        if isinstance(dct, dict):
            all_keys = list(dct.keys())
        else: # it must be a pandas df
            all_keys = dct.columns
        for field in all_keys:
            if field in ["sample_mens", "sample_womens", "sample_newbie", "sample_email_type", "sample_visit", "mens", "womens", "newbie", "email_type", "visit"]:
                binary_map = {"No": 0, "Yes": 1}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_zip_code", "zip_code"]:
                zip_code_map = {"Suburban area": 0, "Rural area": 1, "Urban area": 2}
                dct[field] = zip_code_map[dct[field]]
            elif field in ["sample_channel", "channel"]:
                zip_code_map = {"Phone": 0, "Web": 1, "Multichannel": 2}
                dct[field] = zip_code_map[dct[field]]
            elif field in ["sample_recency", "recency"]:
                recency_map = {"1 - 4": 0, "5 - 8": 1, "9 - 12": 2}
                dct[field] = recency_map[dct[field]]
            elif field in ["sample_history", "history"]:
                history_map = {"0 - 100": 0, "100 - 200": 1, "200 - 350": 2, "350 - 500": 3, "500 - 750": 4, "750 - 1000": 5, "1000 +": 6}
                dct[field] = history_map[dct[field]]
        return dct

    def interpret_samples(self, dct):
        if isinstance(dct, dict):
            all_keys = list(dct.keys())
        else: # it must be a pandas df
            all_keys = dct.columns
        for field in all_keys:
            if field in ["sample_mens", "sample_womens", "sample_newbie", "sample_email_type", "sample_visit", "mens", "womens", "newbie", "email_type", "visit"]:
                binary_map = {0: "No", 1: "Yes"}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_zip_code", "zip_code"]:
                zip_code_map = {0: "Suburban area", 1: "Rural area", 2: "Urban area"}
                dct[field] = zip_code_map[dct[field]]
            elif field in ["sample_channel", "channel"]:
                zip_code_map = {0: "Phone", 1: "Web", 2: "Multichannel"}
                dct[field] = zip_code_map[dct[field]]
            elif field in ["sample_recency", "recency"]:
                recency_map = {0: "1 - 4", 1: "5 - 8", 2: "9 - 12"}
                dct[field] = recency_map[dct[field]]
            elif field in ["sample_history", "history"]:
                history_map = {0: "0 - 100", 1: "100 - 200", 2: "200 - 350", 3: "350 - 500", 4: "500 - 750", 5: "750 - 1000", 6: "1000 +"}
                dct[field] = history_map[dct[field]]
        return dct
    
    def discretize(self, sample_df, hard_filter=True):
        # convert generated samples to a processed dataset
        sample_df = sample_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
        if "history" in sample_df.columns:
            sample_df["history"] = pd.to_numeric(sample_df["history"], errors='coerce')
            sample_df.loc[sample_df["history"] <= 100, "history"] = 0
            sample_df.loc[(sample_df["history"] > 100) & (sample_df["history"] <= 200), "history"] = 1
            sample_df.loc[(sample_df["history"] > 200) & (sample_df["history"] <= 350), "history"] = 2
            sample_df.loc[(sample_df["history"] > 350) & (sample_df["history"] <= 500), "history"] = 3
            sample_df.loc[(sample_df["history"] > 500) & (sample_df["history"] <= 750), "history"] = 4
            sample_df.loc[(sample_df["history"] > 750) & (sample_df["history"] <= 1000), "history"] = 5
            sample_df.loc[sample_df["history"] > 1000, "history"] = 6
            sample_df["history"] = sample_df["history"].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1, 2, 3, 4, 5, 6]))
        if "recency" in sample_df.columns:
            sample_df["recency"] = sample_df["recency"].apply(lambda x: float(x))
            sample_df.loc[sample_df["recency"] <= 4, "recency"] = 0
            sample_df.loc[(sample_df["recency"] > 4) & (sample_df["recency"] <= 8), "recency"] = 1
            sample_df.loc[sample_df["recency"] > 8, "recency"] = 2
            sample_df["recency"] = sample_df["recency"].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1, 2]))
        for field in ["mens", "womens", "newbie", "email_type", "visit"]:
            if field in sample_df.columns:
                field_map = {"No": 0, "Yes": 1}
                sample_df[field] = sample_df[field].replace(field_map)
                sample_df[field] = sample_df[field].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1]))
        if "zip_code" in sample_df:
            field_map = {"Suburban": 0, "Rural": 1, "Urban": 2}
            sample_df["zip_code"] = sample_df["zip_code"].replace(field_map)
            sample_df["zip_code"] = sample_df["zip_code"].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1, 2]))
        if "channel" in sample_df:
            field_map = {"Phone": 0, "Web": 1, "Multichannel": 2}
            sample_df["channel"] = sample_df["channel"].replace(field_map)
            sample_df["channel"] = sample_df["channel"].map(lambda x: x if not np.isnan(x) else np.random.choice([0, 1, 2]))
        return sample_df