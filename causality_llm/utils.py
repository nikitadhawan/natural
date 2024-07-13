import random, os
import numpy as np
from scipy.stats import norm
import torch
    
def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def KL(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64) + 1e-6
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def MSE(a, b):
    return np.mean((a - b)**2)

def drop_feat(feat_dict, drop_prob, X_cols):
    all_feat = list(feat_dict.keys())
    for feat_name in all_feat:
        # delete any key in X_cols wp drop_prob
        if feat_name in X_cols:
            if np.random.sample() < drop_prob:
                del feat_dict[feat_name]
        # delete any key not in X_cols
        else:
            del feat_dict[feat_name]
    return feat_dict

def shuffle_dict(dct):
    lst = list(dct.items())
    np.random.shuffle(lst)
    dct = dict(lst)
    return dct

def qa_interleaved_enum(q_dct, options_dct, a_enum, to_enum):
    all_interleaved_options = []
    alph = ["a) ", "b) ", "c) ", "d) "]
    for option in a_enum:
        interleaved_enum = " \n\n## Questions"
        for num in range(len(to_enum)):
            key = to_enum[num]
            interleaved_enum += " \n\nQ: " + q_dct[key] 
            interleaved_enum += " \nOptions: " 
            for i in range(len(options_dct[key])):
                interleaved_enum += alph[i] + options_dct[key][i] + " "
            split_option = [i.split(":") for i in option.split(",")]
            interleaved_enum += " \nA: " + split_option[num][1][1:]
        all_interleaved_options.append(interleaved_enum)
    return all_interleaved_options


def concatenate_q(dct):
    keys = list(dct.keys())
    num = 1
    all_qs = " \nAnswer the following questions."
    for key in keys:
        all_qs += " \nQ" + str(num) + ": " + dct[key]
        num += 1
    all_qs += "\n"
    return all_qs

def enumerate_strings(dct, string=True):
    keys = list(dct.keys())
    keys.reverse()
    num = len(keys)
    all_enumerated = dct[keys[0]]
    all_enumerated = ["A" + str(num) + ": " + e for e in all_enumerated]
    for key in keys[1:]:
        num -= 1
        cur_len = len(all_enumerated)
        all_enumerated *= len(dct[key])
        for j in range(len(dct[key])):
            all_enumerated[j*cur_len : (j+1)*cur_len] = [dct[key][j] + ", " + e for e in all_enumerated[j*cur_len : (j+1)*cur_len]]
        all_enumerated = ["A" + str(num) + ": " + e for e in all_enumerated]
    return all_enumerated

def enum_to_dcts(enumerated, to_enum):
    return_dcts = []
    for elem in enumerated:
        separate = [i.split(":") for i in elem.split(",")]
        dct = {}
        for field in range(len(to_enum)):
            dct["sample_" + to_enum[field]] = separate[field][1][1:]
        return_dcts.append(dct)
    return return_dcts

def get_sample_text(dct, dataset):
    all_keys = list(dct.keys())
    questions = dataset.get_question_prompt(all_keys)
    dct = dataset.interpret_samples(dct)
    return_text = " \n\n## Questions and their correct answers"
    for key in all_keys:
        return_text += "\nQ: " + questions[key] + " A: " + str(dct[key]) + "."
    return return_text