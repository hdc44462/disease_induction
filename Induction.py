import time
import json
import itertools
import numpy as np
import math
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from sqlalchemy import create_engine
import sys

main_symptom = sys.argv[1].split(',')
minor_symptom = sys.argv[2].split(',')
age = int(sys.argv[3])
month = int(sys.argv[4])
batchno = sys.argv[5]
deathrate = int(sys.argv[6])

histDB = {'host': '10.106.5.205', 'port': 3306, 'user': 'uszchenye', 'password': 'cy629!', 'database': 'MY_ANA_test'}
# batchno = '15380174'


# connect to MYSQL
def sql_extract(dic, sql):
    connect_info = 'mysql+pymysql://' + str(dic['user']) + ':' + str(dic['password']) + '@' + str(
        dic['host']) + ':' + str(dic['port']) + '/' + str(dic['database']) + '?charset=utf8'
    engine = create_engine(connect_info)  # use sqlalchemy to build link-engine
    return pd.read_sql(sql=sql, con=engine)


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
    disease_season = read_json('disease_season.json')
    disease_age = read_json('disease_age.json')
    disease_death = read_json('disease_death.json')

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

    def risk_deathrate(self, rate):
        risk = []
        disease_name = []
        for i in self.disease_death:
            disease_name.append('deathrate_' + self.disease_CNtoENname[i])
            if (rate >= self.disease_death[i]['lowlim']) and (rate <= self.disease_death[i]['upplim']):
                risk.append(0)
            else:
                risk.append(1)
        result = pd.DataFrame([risk], columns=disease_name)
        return result

    def risk_age(self, age):
        risk = []
        disease_name = []
        for i in self.disease_age:
            disease_name.append('age_' + self.disease_CNtoENname[i])
            if (age >= self.disease_age[i]['lowlim']) and (age <= self.disease_age[i]['upplim']):
                risk.append(0)
            else:
                risk.append(1)
        result = pd.DataFrame([risk], columns=disease_name)
        return result

    def risk_season(self, month):
        risk = []
        disease_name = []
        for i in self.disease_season:
            disease_name.append('season_' + self.disease_CNtoENname[i])
            if month in self.disease_season[i]['month']:
                risk.append(0)
            else:
                risk.append(1)
        result = pd.DataFrame([risk], columns=disease_name)
        return result

    # create cpd table    P(disease|[causes]) = exp(sum(-log_risks of each cause))

    def __init__(self, main_symptom, minor_symptom, age, month, deathrate, batchno):
        # main_symptom:主要症状
        # side_symptom:次要症状
        # type_list:基于病理将疾病分类：呼吸道、消化道等
        # sub_cause,sub_symptom：用主要症状对应疾病分类,筛选疾病

        self.main_symptom = main_symptom
        self.minor_symptom = minor_symptom
        self.type_list = self.classif()
        self.sub_cause = [{key: value for key, value in self.cause_disease.items()
                           if key in self.type_list[i]['disease']} for i in self.type_list]
        self.sub_symptom = [{key: value for key, value in self.disease_symptom.items()
                             if key in self.type_list[i]['disease']} for i in self.type_list]
        all_symptom = []
        temp = 0
        for i in main_symptom + minor_symptom:
            if i not in ['腹泻-黄色稀粪', '腹泻-白/灰色稀粪', '腹泻-水样稀粪']:
                all_symptom.append(i)
            else:
                temp += 1
        if temp == 0:
            self.all_symptom = all_symptom
        else:
            self.all_symptom = all_symptom + ['腹泻-黄色稀粪', '腹泻-白/灰色稀粪', '腹泻-水样稀粪']
        self.sample_disease = {}
        season_risk = self.risk_season(month)
        deathrate_risk = self.risk_deathrate(deathrate)
        age_risk = self.risk_age(age)
        swine = sql_extract(histDB, '''select swine_PED,swine_RV,swine_Ecoli,swine_Salm,swine_Clostridiu,swine_PRRS,swine_HPS,swine_SS,
        swine_Pm,swine_PCV,swine_Mhyo,swine_Bb,swine_SIV,swine_HC,swine_PRV,swine_PPV,swine_PDCoV,swine_FMD,swine_SVV,swine_JEV,swine_PPE,
        swine_APP,swine_PTQJ,swine_Ery,swine_Cp,swine_SD,swine_Rr,swine_FHXBT,swine_TP 
        from disease_risk_tb where batchno = \'%s\'''' % batchno + 'limit 1')
        if swine.shape[0] == 0:
            swine = pd.DataFrame([[1] * swine.shape[1]], columns=swine.columns)
        envir = sql_extract(histDB, '''select envir_PED,envir_RV,envir_Ecoli,envir_Salm,envir_Clostridiu,envir_PRRS,envir_HPS,envir_SS,
        envir_Pm,envir_PCV,envir_Mhyo,envir_Bb,envir_SIV,envir_HC,envir_PRV,envir_PPV,envir_PDCoV,envir_FMD,envir_SVV,envir_JEV,envir_PPE,
        envir_APP,envir_PTQJ,envir_Ery,envir_Cp,envir_SD,envir_Rr,envir_FHXBT,envir_TP 
        from fsample_envir_result_info where batchno = \'%s\'''' % batchno + 'limit 1')
        if envir.shape[0] == 0:
            envir = pd.DataFrame([[1] * envir.shape[1]], columns=envir.columns)
        self.obs = pd.concat([season_risk, deathrate_risk, age_risk, swine, envir], axis=1).fillna(1)

    # 基于疾病与症状分类，获得疑似疾病
    def classif(self):
        type_list = []
        for i in self.main_symptom:
            for j in self.symptom_classification:
                if i in self.symptom_classification[j] and j not in type_list:
                    type_list.append(j)
        res = {}
        for i in type_list:
            res[i] = {'disease': [], 'symptom': []}
            for j in self.disease_type:
                if i in self.disease_type[j]:
                    res[i]['disease'].append(j)
            for j in self.main_symptom:
                if j in self.symptom_classification[i]:
                    res[i]['symptom'].append(j)
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
        self.model = model
        return model

    # 使用观察到的症状和搭建好的贝叶斯网络 获得推理结果
    def test_predict(self, sub_cause, sub_symptom, model):
        print('function test_predict start!')
        start_time = time.time()
        cause = list(self.reverse(sub_cause).keys())
        disease = list(sub_cause.keys())
        temp = []
        for i in disease:
            if 'season_' + i in self.obs.columns:
                temp.append('season_' + i)
            if 'age_' + i in self.obs.columns:
                temp.append('age_' + i)
            if 'deathrate_' + i in self.obs.columns:
                temp.append('deathrate_' + i)
            if 'envir_' + i in self.obs.columns:
                temp.append('envir_' + i)
            if 'swine_' + i in self.obs.columns:
                temp.append('swine_' + i)
        other_symptom = []
        all_symptom = self.all_symptom
        for i in self.reverse(sub_symptom).keys():
            if i not in all_symptom:
                other_symptom.append(i)
        for i in self.all_symptom:
            if i not in self.reverse(sub_symptom).keys():
                all_symptom = [j for j in all_symptom if j != i]
        symptom = pd.DataFrame([[1] * len(other_symptom) + [0] * len(all_symptom)], columns=other_symptom + all_symptom)
        obs = pd.concat([symptom, self.obs[temp]], axis=1)
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
        result = dict(pd.concat(temp, axis=1).max(axis=1).sort_values(ascending=False).head(5))
        for i in result.keys():
            disease_name = [key for key, value in self.disease_CNtoENname.items() if value == i][0]
            self.sample_disease[disease_name] = result[i]
        return {key: str(round(value * 100, 2)) + '%' for key, value in result.items()}

    # 症状推荐
    def sampling(self, disease=[]):
        if not disease:
            disease = self.sample_disease
        sample_recommend = {'疾病中文': [], '疾病缩写': [], '剖检': [], '非剖检': []}
        for i in disease:
            sample_recommend['疾病中文'].append(i)
            sample_recommend['疾病缩写'].append(self.disease_CNtoENname[i])
            autopsy = self.sample[i]['剖检']
            biopsy = self.sample[i]['非剖检']
            for j in autopsy:
                if j not in sample_recommend['剖检']: sample_recommend['剖检'].append(j)
            for j in biopsy:
                if j not in sample_recommend['非剖检']: sample_recommend['非剖检'].append(j)
        return sample_recommend


if __name__ == '__main__':
    induction = BayesN(main_symptom, minor_symptom, age, month, deathrate, batchno)
    #     induction = BayesN(['腹胀','腹泻-黄色稀粪','咳嗽'],['突然死亡','鼻甲变形'],20,4,15,'WX310603162020200801')
    disease = induction.predict()
    sample = induction.sampling()
    sys.exit({'induction': disease, 'sample': sample})
