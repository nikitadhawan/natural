import numpy as np
import pandas as pd

class Botox(object):

    def __init__(self, feature_name=None, prop_score=None, train_size=5000, seed=10):
        self.ground_truth = (42 - 1) / 100
        self.seed = seed
        self.train_size = train_size
        self.num_outcomes = 2
        self.subreddits = ["migraine"] 
        self.treatment_names = ["onabotulinumtoxin", "botox", "topiramate", "topamax", "topimax", "epitomax", "topiragen", "eprontia", "qudexy", "trokendi"]
        gpt4_misspelled = ["botulinum", "onabotulimuntoxin", "onabotulinumtoxan", "onabotulinymtoxina", "onabotulimuntoxina", "onabotulinmtoxina", "onabotulinumtxina", "onabotulintmoxina", "onabotulinumtaxin", "onabotulnumtoxina"
                           "botex", "boto", "bottox", "botx", "botoks", "botoxx", "botto", "bottocks", "botoxz", "botoxs",
                           "topiramte", "topiramat", "topriamate", "topiramaet", "topiramtae", "topiramet", "topirameat", "topiramatte", "topiramae", "topiramaet",
                           "topamx", "topamxa", "topamaz", "topamxa", "topmax", "topamaz", "topamx", "toapmax", "topama", "topamxx",
                           "epitomxa", "epitmoax", "epiotmax", "epitomaz", "epitoma", "epitmoax", "epiotomax", "epitomaxx", "epitomxa", "epitomaz",
                           "topirgan", "topiragn", "topirgaen", "topirgne", "topiraegn", "topirgen", "topiragn", "topirgaen", "topirgane", "topiargen",
                           "epronita", "eprontai", "epronti", "epronita", "epronita", "epronia", "eprontai", "eprontiaa", "epronti", "epronta",
                           "qudxy", "qudex", "quedxy", "qudexy", "qudxey", "quedexy", "qudexxy", "quedyx", "qudex", "qudxey",
                           "troknedi", "troekndi", "trokenid", "troekndi", "troknedi", "trokend", "trokindi", "trokedni", "trkendi", "trokenid"]
        perplexity_misspelled = ["topromate", "topirimate", "topiramat",
                                 "topomax",
                                 "epitomaks", "epitomix",
                                 "topiragin", "topiragun", "topiragin", 
                                 "epronia", "eprontea",
                                 "qudexi", "qudexie", "qudexey",
                                 "trokendy", "trokendie", "trokendii"
                                 ]
        self.treatment_names += gpt4_misspelled + perplexity_misspelled
        self.outcome_words = ["stop", "continue", "adverse", "side", "effect", "nausea", "aura", "pound", "tolerate", "quit", "scotoma", "visual", "light", "sound", "eye", "constipation", "muscle spasm", "reaction", "fatigue", "drowsy", "allergic", "cognitive", "memory", "nausea", "weight", "kidney", "days", "MMD"]

    def XYZ_names(self, return_enum_cols=False, return_strat_cols=False):
        X_cols = ["age", "sex", "country", "baseline_MMD", "final_MMD", "duration_days", "dosage", "adverse_effect"]
        Y_cols = "discontinued"
        Z_cols = "drug_type"
        if return_enum_cols or return_strat_cols: # we don't enumerate for real datasets
            strat_cols = ["age", "sex", "country", "baseline_MMD", "duration_days"]
            return X_cols, Y_cols, Z_cols, strat_cols
        return X_cols, Y_cols, Z_cols
    
    def hard_filter_ty(self, extract):
        extract = extract.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
        treatment = extract['drug_type'].apply(lambda x: x if x in ['Topiramate', 'Topiragen', 'Topamax', 'Topomax', 'Topimax', 'Trokendi', 'Trokenidi', 'Topirimate', 'Qudexy', 'Topiramat', 'Botox'] else np.nan).dropna()
        extract = extract.loc[treatment.index]
        outcome = extract['discontinued'].notnull()
        extract = extract[outcome]
        return extract
    
    def hard_filter_inclusion(self, extract):
        extract.loc[:, "inclusion_score"] = 0 
        age = (extract['age']>=18) | (extract['age']<=65) | (extract['age'].isna())
        baseline_MMD = ((extract['baseline_MMD'] >= 15) & (extract['baseline_MMD'] <= 30)) | (extract['baseline_MMD'].isna())
        botox = (extract["drug_type"] == 0) 
        topiramate = (extract["drug_type"] == 1) & (extract["dosage"] <= 100)
        dosage = botox | topiramate | (extract['dosage'].isna())
        for criterion in [baseline_MMD, dosage, age]:
            extract.loc[criterion, "inclusion_score"] += 1
        known_to_unmatch = extract[extract["inclusion_score"]<3].index
        extract = extract.drop(known_to_unmatch)
        extract = extract.drop(["dosage"], axis=1)
        return extract
    
    def get_options(self, feature_names):
        options = {
            "age": ["No", "Yes"], 
            "sex": ["Male", "Female"],
            "country": ['United States', 'Canada', 'Australia', 'Europe', "UK", 'Germany', 'Argentina', 'Scotland', "Brazil"],
            "baseline_MMD": ["No", "Yes"], 
            "duration_days": ["Upto 30 days", "Over 30 days"],
            "drug_type": ["Botox injection", "Topiramate"], # treatment
            "discontinued": ["No", "Yes"] # outcome
        } 
        return {key: options[key] for key in feature_names} 


    def get_question_prompt(self, feature_names): 
        questions = {
            "age": "Is the user's age greater than 25 years?",
            "sex": "What is the reported sex of the user?",
            "country": "Which country does the user reside in?",
            "baseline_MMD": "Did the user experience migraine symptoms for greater than 15 days per month, before starting treatment?",
            "duration_days": "For what duration did the user take the treatment, before reporting their experience?", 
            "drug_type": "Which treatment did the user take?", # treatment
            "discontinued": "Did the user discontinue the treatment above due to adverse side effects?" # outcome
        } 
        return {key: questions[key] for key in feature_names}

    def transform_samples(self, dct):
        if isinstance(dct, dict):
            all_keys = list(dct.keys())
        else: # it must be a pandas df
            all_keys = dct.columns
        for field in all_keys:
            if field in ["sample_drug_type", "drug_type"]:
                binary_map = {"Botox injection": 0, "Topiramate": 1}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_discontinued", "discontinued", "sample_age", "age", "sample_baseline_MMD", "baseline_MMD"]:
                binary_map = {"No": 0, "Yes": 1}
                dct[field] = binary_map[dct[field]] 
            elif field in ["sample_sex", "sex"]:
                binary_map = {"Male": 0, "Female": 1}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_country", "country"]:
                country_map = {'United States': 0, 'US': 0, 'USA': 0, 'Ohio': 0, 'California': 0, 'PNW': 0, 'British Columbia, Canada': 1, 'Canada': 1, 'Australia': 2, 
                               'Europe': 3, "EU": 3, "UK": 4, "United Kingdom": 4, 'Germany': 5, 'Argentina': 6, 'Scotland': 7, "Brazil": 8}
                dct[field] = country_map[dct[field]]
            elif field in ["sample_duration_days", "duration_days"]:
                duration_map = {"Upto 30 days": 0, "Over 30 days": 1}
                dct[field] = duration_map[dct[field]]
        return dct

    def interpret_samples(self, dct):
        if isinstance(dct, dict):
            all_keys = list(dct.keys())
        else: # it must be a pandas df
            all_keys = dct.columns
        for field in all_keys:
            if field in ["sample_sex", "sex"]:
                sex_map = {0: "Male", 1: "Female"}
                dct[field] = sex_map[dct[field]]
            elif field in ["sample_country", "country"]:
                country_map = {0: 'United States', 1: 'Canada', 2: 'Australia', 3: 'Europe', 4: "United Kingdom", 
                               5: 'Germany', 6: 'Argentina', 7: 'Scotland', 8: "Brazil"}
                dct[field] = country_map[dct[field]]
            elif field in ["sample_drug_type", "drug_type"]:
                drug_map = {0: "Botox injection", 1: "Topiramate"}
                dct[field] = drug_map[dct[field]]
            elif field in ["sample_discontinued", "discontinued", "sample_age", "age", "sample_baseline_MMD", "baseline_MMD"]:
                binary_map = {0: "No", 1: "Yes"}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_duration_days", "duration_days"]:
                duration_map = {0: "Upto 30 days", 1: "Over 30 days"}
                dct[field] = duration_map[dct[field]]
        return dct

    def discretize(self, sample_df, hard_filter=True, inf=True):
        sample_df = sample_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
        if hard_filter:
            sample_df = self.hard_filter_ty(sample_df)
        sex_map = {"Male": 0, "Female": 1}
        sample_df["sex"] = sample_df["sex"].replace(sex_map)
        country_map = {'United States': 0, 'US': 0, 'USA': 0, 'Ohio': 0, 'California': 0, 'PNW': 0, 'British Columbia, Canada': 1, 'Canada': 1, 'Australia': 2, 
                       'Europe': 3, "EU": 3, "UK": 4, "United Kingdom": 4, 'Germany': 5, 'Argentina': 6, 'Scotland': 7, "Brazil": 8}
        sample_df["country"] = sample_df["country"].replace(country_map)    
        for numeric_feat in ['age', 'sex', 'country', 'baseline_MMD', 'final_MMD', 'duration_days', 'dosage']:
            if numeric_feat in sample_df.columns:
                sample_df[numeric_feat] = pd.to_numeric(sample_df[numeric_feat], errors='coerce')
        for field in ["discontinued"]:
            if field in sample_df.columns:
                field_map = {"No": 0, "Yes": 1}
                sample_df[field] = sample_df[field].replace(field_map)

        sample_df.loc[sample_df["drug_type"].isin(['Botox']), "drug_type"] = 0 
        sample_df.loc[sample_df["drug_type"].isin(['Topiramate', 'Topiragen', 'Topamax', 'Topomax', 'Topimax', 'Trokendi', 'Topirimate', 'Qudexy', 'Topiramat']), "drug_type"] = 1
        sample_df = sample_df.drop(sample_df[~sample_df["drug_type"].isin([0,1])].index)

        sample_df["drug_type"] = sample_df["drug_type"].astype('int8')
        sample_df["discontinued"] = sample_df["discontinued"].astype('int8')
        
        if hard_filter:
            sample_df = self.hard_filter_inclusion(sample_df)
        elif inf: 
            sample_df.loc[sample_df["age"] <= 25, "age"] = 0
            sample_df.loc[sample_df["age"] > 25, "age"] = 1
            sample_df.loc[sample_df["baseline_MMD"] <= 15, "baseline_MMD"] = 0
            sample_df.loc[sample_df["baseline_MMD"] > 15, "baseline_MMD"] = 1
            sample_df.loc[sample_df["duration_days"] <= 30, "duration_days"] = 0
            sample_df.loc[sample_df["duration_days"] > 30, "duration_days"] = 1
        sample_df = sample_df.reset_index(drop=True)
        return sample_df