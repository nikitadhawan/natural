import numpy as np
import pandas as pd

class Liraglutide(object):

    def __init__(self, feature_name=None, prop_score=None, train_size=5000, seed=10):
        self.ground_truth = (4.4 - 19.1) / 100
        self.seed = seed
        self.train_size = train_size
        self.num_outcomes = 2
        self.subreddits = ["liraglutide"] # ["Mounjaro", "Ozempic", "fasting", "intermittentfasting", "keto", "loseit", "Semaglutide", "SuperMorbidlyObese", "PlusSize"]
        self.treatment_names = ["ozempic", "saxenda", "semaglutide", "liraglutide", "wegovy", "rybelsus", "victoza"]
        perplexity_misspelled = ["ozenpic", "osempic", "ozempik", 
                                 "saxend",
                                "semaglitude", "semagluteid", 
                                "liraglutid", 
                                "wegovi", "wegoby",
                                "rybelsis", "ribelsus", "rybelzus",
                                "victoz",
                                ]
        gpt4_misspelled = ["ozempci", "ozmepic", "ozepmic", "ozempik", "ozemipic", "ozenpic", "osemepic", "ozemepic", "ozeempic", "ozempec", 
                          "sexenda", "saxanda", "saxeeda", "sacxenda", "sxenda", "saxend", "sxaenda", "saxendaa", "saxeden",
                          "semaglutid", "semaglutied", "seamaglutide", "semaglutde", "semagluide", "semagltuide", "semaglutidde", "semaglutite", "smeaglutide", "semagulitde",
                          "liragutide", "liraglute", "liarglutide", "liragultide", "leraglutide", "liragludite", "liraglitide", "liraglutid", "liraglutade", "liragluetide",
                          "wegovi", "wegov", "wegoyv", "wegvo", "wegovy", "wegovy", "wegovuy", "wegoy", "wegovoy", "wegovy",
                          "ryblesus", "rybelsu", "rybeslus", "rybelus", "rybelsuss", "rybelsis", "rybelssus", "rybelssu", "rybelus", "ryblessus",
                          "victoz", "vicotza", "victosa", "victoaz", "victoa", "victza", "victosa", "vicotza", "victzoa", "victzo"
                          ]
        self.treatment_names += gpt4_misspelled + perplexity_misspelled
        self.outcome_words = ["kg", "kilo", "lb", "pound", "weigh", "drop", "loss", "lost", "gain", "hb", "a1c", "hemoglobin", "haemoglobin", "glucose", "sugar"]

    def XYZ_names(self, return_enum_cols=False, return_strat_cols=False):
        X_cols = ["age", "sex", "country", "t2dm", "metformin", "bmi",
                  "start_HbA1c", "end_HbA1c", "start_weight", "end_weight", 
                  "weight_unit", "weight_change", "percentage_weight_change", 
                  "duration_days", "dosage"]
        Y_cols = "target_achieved"
        Z_cols = "drug_type"
        if return_enum_cols or return_strat_cols: # we don't enumerate for real datasets
            strat_cols = ["age", "sex", "bmi", "start_HbA1c", "start_weight", "duration_days"]
            return X_cols, Y_cols, Z_cols, strat_cols
        return X_cols, Y_cols, Z_cols
    
    def hard_filter_ty(self, extract):
        extract = extract.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
        treatment = extract['drug_type'].apply(lambda x: x if x in ['Semaglutide', 'Wegovy', 'Liraglutide', 'Ozempic (Semaglutide)', 'Ozempic', 'Rybelsus', 'Saxenda (Liraglutide)', 'Wegovy/Ozempic', 'Wegovy/Semaglutide', 'Saxenda',
                                                                    'Wegovy or Ozempic (Semaglutide)', 'Victoza', 'Wegovy/Ozempic (Semaglutide)', 'Wegovy (Same drug as Ozempic)', 'Ozempic/Wegovy (Semaglutide)', 
                                                                    'Wegovy (Semaglutide)', 'Saxenda or Victoza', 'Ozempic/Wegovy', 'WeGovy', 'Rybelsus (Semaglutide)', 'Wegovy 2.4 (Semaglutide)', 'Rybelsus 14 mg', 'Wegovy - Semaglutide'] 
                                                                    else np.nan).dropna()
        extract = extract.loc[treatment.index]
        outcome = (extract['start_weight'].notnull()) & ((extract['end_weight'].notnull()) | (extract['weight_change'].notnull()) | (extract['percentage_weight_change'].notnull()))
        extract = extract[outcome]

        return extract
    
    def hard_filter_inclusion(self, extract):
        extract.loc[:, "inclusion_score"] = 0 
        t2dm = ((extract['t2dm']==1) | (extract['t2dm'].isna())) & (((extract['start_HbA1c'] >= 7) & (10.5 >= extract['start_HbA1c'])) | (extract['start_HbA1c'].isna())) 
        metformin = (extract['metformin']==1) | (extract['metformin'].isna())
        dosage = ((extract["drug_type"] == 0) & (extract["dosage"] == 1)) | ((extract["drug_type"] == 1) & (extract["dosage"] == 1.2)) | (extract['dosage'].isna())
        for criterion in [t2dm, metformin, dosage]:
            extract.loc[criterion, "inclusion_score"] += 1
        known_to_unmatch = extract[extract["inclusion_score"]<3].index
        extract = extract.drop(known_to_unmatch)
        extract = extract.drop(["t2dm", "metformin", "dosage"], axis=1)
        return extract

    
    def get_options(self, feature_names):
        options = {
            "age": ["No", "Yes"], 
            "sex": ["Male", "Female"], 
            "country": ["United States", "Mexico", "Canada", "Australia", "United Kingdom", "Belgium", "Greece", 
                        "Germany", "Brazil", "Netherlands", "Colombia", "Denmark", "Finland", "Middle East", "Norway", "Serbia"],
            "bmi": ["No", "Yes"], 
            "start_HbA1c": ["No", "Yes"], 
            "start_weight": ["No", "Yes"], 
            "duration_days": ["Upto 120 days", "Over 120 days"],
            "drug_type": ["Semaglutide like Ozempic or Wegovy or Rybelsus", "Liraglutide like Saxenda or Victoza"], # treatment
            "target_achieved": ["No", "Yes"] # outcome
        } 
        return {key: options[key] for key in feature_names}


    def get_question_prompt(self, feature_names): 
        questions = {
            "age": "Is the user's age greater than 45 years?",
            "sex": "What is the reported sex of the user?",
            "t2dm": "Does the user have Type 2 Diabetes?",
            "metformin": "Does the user mention taking Metformin or other background medication?",
            "bmi": "Is the user's Body Mass Index (BMI) greater than 28.5 kg per meter squared?",
            "start_HbA1c": "Is the user's starting HbA1c level before treatment greater than 7.5%?",
            "start_weight": "Is the user's starting weight level before treatment greater than 220 lbs?",
            "end_HbA1c": "What is the user's final HbA1c level at the end of their treatment regime?",
            "end_weight": "What is the user's final weight at the end of their treatment regime, in lb?",
            "weight_unit": "What units was the user's weight originally reported in?",
            "weight_change": "What is the net change in the user's weight, in lb?", 
            "percentage_weight_change": "What is the percentage change in the user's weight?", 
            "duration_days": "For what duration did the user take the treatment, before reporting change in their weight?", 
            "dosage": "What was the dosage of the treatment taken by the user, in mg?",
            "drug_type": "Which treatment did the user take?", # treatment
            "target_achieved": "Did the user lose 10 or more percent of their initial weight?" # outcome
        } 
        return {key: questions[key] for key in feature_names}

    def transform_samples(self, dct):
        if isinstance(dct, dict):
            all_keys = list(dct.keys())
        else: # it must be a pandas df
            all_keys = dct.columns
        for field in all_keys:
            if field in ["sample_drug_type", "drug_type"]:
                binary_map = {"Semaglutide like Ozempic or Wegovy or Rybelsus": 0, "Liraglutide like Saxenda or Victoza": 1}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_target_achieved", "sample_age", "sample_bmi", "sample_start_HbA1c", "sample_start_weight", "target_achieved", "age", "bmi", "start_HbA1c", "start_weight"]:
                binary_map = {"No": 0, "Yes": 1}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_sex", "sex"]:
                binary_map = {"Male": 0, "Female": 1}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_country", "country"]:
                country_map = {"United States": 0, "Mexico": 1, "Canada": 2, "Australia": 3, "United Kingdom": 4, "Belgium": 5, "Greece": 6, "Germany": 7, "Brazil": 8, 
                               "Netherlands": 9, "Colombia": 10, "Denmark": 11, "Finland": 12, "Middle East": 13, "Norway": 14, "Serbia": 15}
                dct[field] = country_map[dct[field]]
            elif field in ["sample_duration_days", "duration_days"]:
                duration_map = {"Upto 120 days": 0, "Over 120 days": 1}
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
            elif field in ["sample_t2dm", "t2dm", "sample_metformin", "metformin", "sample_target_achieved", "target_achieved",
                           "sample_age", "sample_bmi", "sample_start_HbA1c", "sample_start_weight", "target_achieved", "age", "bmi", "start_HbA1c", "start_weight"]:
                binary_map = {0: "No", 1: "Yes"}
                dct[field] = binary_map[dct[field]]
            elif field in ["sample_drug_type", "drug_type"]:
                drug_map = {0: "Semaglutide like Ozempic or Wegovy or Rybelsus", 1: "Liraglutide like Saxenda or Victoza"}
                dct[field] = drug_map[dct[field]]
            elif field in ["sample_country", "country"]:
                country_map = {0: "United States", 1: "Mexico", 2: "Canada", 3: "Australia", 4: "United Kingdom", 5: "Belgium", 6: "Greece", 7: "Germany", 8: "Brazil", 
                               9: "Netherlands", 10: "Colombia", 11: "Denmark", 12: "Finland", 13: "Middle East", 14: "Norway", 15: "Serbia"}
                dct[field] = country_map[dct[field]]
            elif field in ["sample_duration_days", "duration_days"]:
                duration_map = {0: "Upto 120 days", 1: "Over 120 days"}
                dct[field] = duration_map[dct[field]]
        return dct
    
    def discretize(self, sample_df, hard_filter=True, inf=True):
        sample_df = sample_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
        if hard_filter:
            sample_df = self.hard_filter_ty(sample_df)
        sex_map = {"Male": 0, "Female": 1}
        sample_df["sex"] = sample_df["sex"].replace(sex_map)
        country_map = {'United States': 0, "USA": 0, 'Mexico': 1, 'Canada': 2, 'Australia': 3, 'United Kingdom': 4, "UK": 4, 'Belgium': 5, 'Greece': 6, 'Germany': 7, "Brazil": 8, 
                       "Netherlands": 9, "Colombia": 10, "Denmark": 11, "Finland": 12, "Middle East": 13, "Norway": 14, "Serbia": 15}
        sample_df["country"] = sample_df["country"].replace(country_map)
        for numeric_feat in ['age', 'sex', 'country', 'bmi', 'start_HbA1c', 'end_HbA1c', 'start_weight', 'end_weight', 'weight_change', 'percentage_weight_change', 'duration_days', 'dosage']:
            if numeric_feat in sample_df.columns:
                sample_df[numeric_feat] = pd.to_numeric(sample_df[numeric_feat], errors='coerce')
        for field in ["t2dm", "metformin"]:
            if field in sample_df.columns:
                field_map = {"No": 0, "Yes": 1}
                sample_df[field] = sample_df[field].replace(field_map)
        
        if "weight_unit" in sample_df.columns:
            weight_in_kg = sample_df[sample_df['weight_unit']=='kg'].index
            sample_df.loc[weight_in_kg, "start_weight"] *= 2.20462
            sample_df.loc[weight_in_kg, "end_weight"] *= 2.20462
            sample_df.loc[weight_in_kg, "weight_change"] *= 2.20462
            sample_df = sample_df.drop("weight_unit", axis=1)
        
        # different ways of inferring target achievement
        compute_change = sample_df[(sample_df.percentage_weight_change.isna()) & (sample_df.weight_change.isna())].index # our filtering ensures that start+end weight are non-NA for these
        sample_df.loc[compute_change, "weight_change"] = sample_df.loc[compute_change, "end_weight"] - sample_df.loc[compute_change, "start_weight"] 
        compute_target = sample_df[sample_df.percentage_weight_change.isna()].index
        sample_df.loc[compute_target, "percentage_weight_change"] = 100 * (sample_df.loc[compute_target, "weight_change"] / sample_df.loc[compute_target, "start_weight"])
        sample_df.loc[sample_df["percentage_weight_change"] <= -10, "target_achieved"] = 1
        sample_df.loc[~(sample_df["percentage_weight_change"] <= -10), "target_achieved"] = 0

        # this covers all datapoints when gpt returns different treatment names
        sample_df.loc[sample_df["drug_type"].isin(["Ozempic", "Wegovy", "Rybelsus", "Semaglutide"]), "drug_type"] = 0 
        sample_df.loc[sample_df["drug_type"].isin(["Victoza", "Saxenda", "Liraglutide"]), "drug_type"] = 1
        sample_df = sample_df.drop(sample_df[~sample_df["drug_type"].isin([0,1])].index)

        sample_df["drug_type"] = sample_df["drug_type"].astype('int8')
        sample_df["target_achieved"] = sample_df["target_achieved"].astype('int8')

        if hard_filter:
            sample_df = self.hard_filter_inclusion(sample_df)
        elif inf: # samples are already filtered; we need to discretize for causal inference
            sample_df.loc[sample_df["age"] <= 45, "age"] = 0
            sample_df.loc[sample_df["age"] > 45, "age"] = 1
            sample_df.loc[sample_df["bmi"] <= 28.5, "bmi"] = 0 
            sample_df.loc[sample_df["bmi"] > 28.5, "bmi"] = 1
            sample_df.loc[sample_df["start_HbA1c"] <= 7.5, "start_HbA1c"] = 0 # 7.8
            sample_df.loc[sample_df["start_HbA1c"] > 7.5, "start_HbA1c"] = 1
            sample_df.loc[sample_df["start_weight"] <= 220, "start_weight"] = 0
            sample_df.loc[sample_df["start_weight"] > 220, "start_weight"] = 1
            sample_df.loc[sample_df["duration_days"] <= 120, "duration_days"] = 0
            sample_df.loc[sample_df["duration_days"] > 120, "duration_days"] = 1
        sample_df = sample_df.reset_index(drop=True)
        return sample_df
    