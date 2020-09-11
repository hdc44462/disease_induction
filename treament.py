import pandas as pd
import json
import sys
import warnings

warnings.filterwarnings("ignore")

# segment = 13907
# weight = 100
# amount = 30
# disease = 'SIV'

segmentID = int(sys.argv[1])
average_weight = int(sys.argv[2])
total_amount = int(sys.argv[3])
can_disease = sys.argv[4]


# export dict from json

def read_json(filename):
    with open(filename, 'r') as file:
        load_dict = json.load(file)
        return load_dict


def treatment(segment, weight, amount, disease):
    result = []
    result = disease_treatment[(disease_treatment.工段 == segment) & (disease_treatment.病原.apply(lambda x: disease in x))]
    result.iloc[:, 6] = result.iloc[:, 6] * amount * weight
    return result.sort_values('成本')


if __name__ == '__main__':
    if average_weight == 0:
        if segmentID == 13905:
            average_weight= 5
        elif segmentID == 13906:
            average_weight = 15
        elif segmentID == 13907:
            average_weight = 70
        elif segmentID in (13908,13909,13910):
            average_weight = 120
    if total_amount == 0 :
        total_amount = 10
    disease_treatment = pd.read_excel('treatment.xlsx')
    disease_control = read_json('disease_control.json')
    treat = treatment(segmentID, average_weight, total_amount, can_disease)
    if can_disease in disease_control.keys():
        control = disease_control[can_disease]
    else:
        control = []
    sys.exit({'treatment': treat, 'control': control})
