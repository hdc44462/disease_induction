import time
import json
import itertools
import numpy as np
import math
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


# export dict from json
def read_json(filename):
    with open(filename, 'r') as file:
        load_dict = json.load(file)
        return load_dict


class BayesN:
    disease_CNtoENname = read_json('disease_CNtoENname.json')
    cause_disease = read_json('cause_disease.json')
    disease_symptom = read_json('disease_symptom.json')
    base_symptom = read_json('base_symptom.json')
    base_disease = read_json('base_disease.json')
    symptom_classification = read_json('symptom_classification.json')
    disease_type = read_json('disease_type.json')
    sample = read_json('sample.json')

    @staticmethod
    def reverse(dictionary):
        reverse_dictionary = {}
        for i in dictionary:
            for j in dictionary[i]:
                if j not in reverse_dictionary:
                    reverse_dictionary[j] = {i: dictionary[i][j]}
                else:
                    reverse_dictionary[j][i] = dictionary[i][j]
        return reverse_dictionary

    # create cpd table    P(symptom|[disease]) = max(P(symptom|disease))
    @staticmethod
    def create_cpd_upplim(disease_table, base):
        print('function create_cpd_upplim start!')
        start_time = time.time()
        res = {}
        for i in disease_table:
            if i not in res:
                res[i] = []
                for j in itertools.product([1, 0], repeat=len(disease_table[i])):
                    temp = (np.array(list(disease_table[i].values())) * np.array(j)).max()
                    if temp == 0:
                        res[i].append(base[i])
                    else:
                        res[i].append(temp)
        print('function create_cpd_upplim take %d seconds' % (time.time() - start_time))
        return res

    # create cpd table    P(disease|[causes]) = exp(sum(-log_risks of each cause))
    @staticmethod
    def create_cpd_risk(disease_table, base):
        print('function create_cpd_risk start!')
        start_time = time.time()
        res = {}
        for i in disease_table:
            if i not in res:
                res[i] = []
                for j in itertools.product([1, 0], repeat=len(disease_table[i])):
                    temp = (np.array(list(disease_table[i].values())) * np.array(j)).sum()
                    res[i].append(base[i] / math.exp(-temp))
        print('function create_cpd_risk take %d seconds' % (time.time() - start_time))
        return res

    def __init__(self, main_symptom, minor_symptom):
        # main_symptom:主要症状
        # side_symptom:次要症状
        # type_list:基于病理将疾病分类：呼吸道、消化道等
        # sub_cause,sub_symptom：用主要症状对应疾病分类,筛选疾病

        self.main_symptom = main_symptom
        self.all_symptoms = main_symptom + minor_symptom
        self.type_list = self.classif()
        self.sub_cause = [{key: value for key, value in self.cause_disease.items() if key in self.type_list[i]} for i in
                          self.type_list]
        self.sub_symptom = [{key: value for key, value in self.disease_symptom.items()
                             if key in self.type_list[i]} for i in self.type_list]
        self.sample_disease = []

    # 基于疾病与症状分类，获得疑似疾病
    def classif(self):
        type_list = []
        for i in self.main_symptom:
            for j in self.symptom_classification:
                if i in self.symptom_classification[j] and j not in type_list:
                    type_list.append(j)
        res = {}
        for i in type_list:
            temp = []
            for j in self.disease_type:
                if i in self.disease_type[j]:
                    temp.append(j)
            res[i] = temp
        return res

    # 搭建贝叶斯网络
    def create_bayes(self, sub_cause, sub_symptom):
        print('function create_bayes start!')
        start_time = time.time()
        nodes = []
        for i in sub_cause:
            for j in sub_cause[i]:
                nodes.append((j, i))
        for i in sub_symptom:
            for j in sub_symptom[i]:
                nodes.append((i, j))
        model = BayesianModel(nodes)
        cpd_symptom = self.create_cpd_upplim(self.reverse(sub_symptom), self.base_symptom)
        cpd_disease = self.create_cpd_risk(sub_cause, self.base_disease)
        for j in cpd_symptom:
            evi = self.reverse(sub_symptom)[j].keys()
            model.add_cpds(
                TabularCPD(variable=j, variable_card=2, values=[cpd_symptom[j], [1 - k for k in cpd_symptom[j]]],
                           evidence=evi, evidence_card=[2] * len(evi)))
        for j in sub_cause:
            evi = sub_cause[j].keys()
            model.add_cpds(
                TabularCPD(variable=j, variable_card=2, values=[cpd_disease[j], [1 - k for k in cpd_disease[j]]],
                           evidence=evi, evidence_card=[2] * len(evi)))

        for j in self.reverse(sub_cause).keys():
            model.add_cpds(TabularCPD(variable=j, variable_card=2, values=[
                [0],
                [1]
            ]))
        print('function create_bayes take %d seconds' % (time.time() - start_time))
        return model

    # 使用观察到的症状和搭建好的贝叶斯网络 获得推理结果
    def test_predict(self, sub_cause, sub_symptom, model):
        print('function test_predict start!')
        start_time = time.time()
        cause = list(self.reverse(sub_cause).keys())
        symptom = []
        for i in self.reverse(sub_symptom).keys():
            if i not in self.all_symptoms:
                symptom.append(i)
        obs = cause + symptom
        obs = pd.DataFrame([[1] * len(obs) + [0] * len(self.all_symptoms)], columns=obs + self.all_symptoms)
        result = model.predict_probability(obs)
        disease_candidates = result.columns[::2]
        result = result[disease_candidates]
        result.columns = [i[:-2] for i in disease_candidates]
        print('function test_predict take %d seconds' % (time.time() - start_time))
        return result.transpose()

    # 整体推理function
    def predict(self):
        print('function test_predict start!')
        temp = []
        for i in range(len(self.type_list)):
            print('第%d次开始,种类为 %s' % (i, list(self.type_list.keys())[i]))
            sub_cause = self.sub_cause[i]
            sub_symptom = self.sub_symptom[i]
            model = self.create_bayes(sub_cause, sub_symptom)
            temp.append(self.test_predict(sub_cause, sub_symptom, model))
        self.sample_disease = dict(pd.concat(temp, axis=1).max(axis=1).sort_values(ascending=False).head(5))
        return pd.concat(temp, axis=1).max(axis=1).sort_values(ascending=False)

    # 症状推荐
    def sampling(self, disease=[]):
        if not disease:
            disease = self.sample_disease
        disease_ENtoCNname = {value: key for key, value in self.disease_CNtoENname.items()}
        sample_recommend = {'疾病': [], '剖检': [], '非剖检': []}
        for i in disease:
            sample_recommend['疾病'].append(i)
            autopsy = self.sample[disease_ENtoCNname[i]]['剖检']
            biopsy = self.sample[disease_ENtoCNname[i]]['非剖检']
            for j in autopsy:
                if j not in sample_recommend['剖检']: sample_recommend['剖检'].append(j)
            for j in biopsy:
                if j not in sample_recommend['非剖检']: sample_recommend['非剖检'].append(j)
        return sample_recommend


if __name__ == '__main__':
    a = BayesN(['腹泻-白/灰色稀粪', '咳嗽'], ['呕吐', '突然死亡'])
    res = a.predict()
    sam = a.sampling()