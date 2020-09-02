import json
import itertools
import numpy as np
import math
import pandas as pd


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

    @staticmethod
    def create_cpd_upplim(disease_table, base):
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
        return res

    @staticmethod
    def create_cpd_risk(disease_table, base):
        res = {}
        for i in disease_table:
            if i not in res:
                res[i] = []
                for j in itertools.product([1, 0], repeat=len(disease_table[i])):
                    temp = (np.array(list(disease_table[i].values())) * np.array(j)).sum()
                    res[i].append(base[i] / math.exp(-temp))
        return res

    def __init__(self, main_symptom, side_symptom):
        self.main_symptom = main_symptom
        self.all_symptoms = main_symptom + side_symptom
        self.type_list = self.classif()

    #         self.sample_recom = self.sample_recom()
    #         self.med_treat = self.treat_recom()
    def classif(self):
        type_list = []
        for i in self.main_symp:
            for j in self.symptom_classification:
                if i in self.symptom_classification[j] and j not in type_list:
                    type_list.append(j)
        res = []
        for i in type_list:
            temp = []
            for j in self.disease_type:
                if i in self.disease_type[j]:
                    temp.append(j)
            res[i] =temp
        return res

    @staticmethod
    def create_nodes(cause, symptom):
        nodes = []
        for i in cause:
            for j in cause[i]:
                nodes.append((j, i))
        for i in symptom:
            for j in symptom[j]:
                nodes.append((i, j))
        return BayesianModel(nodes)

    def add_probs(self):
        cpd_symptom = create_cpd_upplim(reverse(cpd_symptom), base)
        cpd_disease = create_cpd_risk(disease_table, base)
        for i in cpd_symptom:
            evi = list(self.symptom[self.symptom.symptom == i].disease)
            self.model.add_cpds(
                TabularCPD(variable=i, variable_card=2, values=[cpd_symptom[i], [1 - j for j in cpd_symptom[i]]],
                           evidence=evi, evidence_card=[2] * len(evi)))

        for i in cpd_disease:
            evi = list(self.cause[self.cause.disease == i].cause)
            #             print(i,evi)
            self.model.add_cpds(
                TabularCPD(variable=i, variable_card=2, values=[cpd_disease[i], [1 - j for j in cpd_disease[i]]],
                           evidence=evi, evidence_card=[2] * len(evi)))
        #         fillup all the cpds for cause_nodes(meaningless in calculation cuz those nodes are required information, no update needed)
        for i in self.cause.cause.drop_duplicates():
            print('staaaa')
            if i != 'PRRS':
                self.model.add_cpds(TabularCPD(variable=i, variable_card=2,
                                               values=[
                                                   [0],
                                                   [1]
                                               ]))

    def create_BaysianN(self):
        self.sub_cause = [{key: value for key, value in cause_disease.items() if key in i} for i in self.type_list]
        self.sub_symptom = [{key: value for key, value in disease_symptom.items() if key in i} for i in self.type_list]
        model = create_nodes(sub_cause, sub_symptom)
        self.add_cpds(model)
        return model

    def predict(self):
        sub_cause = [{key: value for key, value in cause_disease.items() if key in i} for i in self.type_list]
        sub_symptom = [{key: value for key, value in disease_symptom.items() if key in i} for i in self.type_list]
        model = create_nodes(sub_cause, sub_symptom)
        self.add_cpds()

        cause_disease = []
        disease_symptom = []
        for i in cause_d:
            for j in list(cause_d[i].keys()):
                cause_disease.append((j, i))
        self.cause = pd.DataFrame(cause_disease).rename({0: 'cause', 1: 'disease'}, axis=1)
        for i in disease_s:
            for j in list(disease_s[i].keys()):
                disease_symptom.append((i, j))
        nodes = cause_disease

        #         get all the nodes for baysian network
        nodes.extend(disease_symptom)
        #         set up the model frame
        print('nonodd')
        self.model = BayesianModel(nodes)
        #         list of disease_to_symptom
        self.symptom = pd.DataFrame(disease_symptom).rename({0: 'disease', 1: 'symptom'}, axis=1)
        self.n_symptom = self.symptom.symptom.drop_duplicates().shape[0]
        #         fillup all the cpds for symptom-nodes
        #         print(self.symptom)
        #         print(disease_s)
        for i in cpd_symptom:
            evi = list(self.symptom[self.symptom.symptom == i].disease)
            #             print(i,evi)
            #             print(evi)
            self.model.add_cpds(
                TabularCPD(variable=i, variable_card=2, values=[cpd_symptom[i], [1 - j for j in cpd_symptom[i]]],
                           evidence=evi, evidence_card=[2] * len(evi)))
        #         fillup all the cpds for disease-nodes
        for i in cpd_disease:
            evi = list(self.cause[self.cause.disease == i].cause)
            #             print(i,evi)
            self.model.add_cpds(
                TabularCPD(variable=i, variable_card=2, values=[cpd_disease[i], [1 - j for j in cpd_disease[i]]],
                           evidence=evi, evidence_card=[2] * len(evi)))
        #         fillup all the cpds for cause_nodes(meaningless in calculation cuz those nodes are required information, no update needed)
        for i in self.cause.cause.drop_duplicates():
            print('staaaa')
            if i != 'PRRS':
                self.model.add_cpds(TabularCPD(variable=i, variable_card=2,
                                               values=[
                                                   [0],
                                                   [1]
                                               ]))

    #         print(self.model.check_model())
    def predict(self, date, age, number, fieldID, symp):
        self.symp = symp
        if number < 10:
            self.method = ['both']
        else:
            self.method = ['group']
        if (field.fieldname == fieldID).sum() != 0:
            field_info = field[field.fieldname == fieldID].drop('fieldname', axis=1)
        else:
            field_info = pd.DataFrame([[1] * (field.shape[1] - 1)], columns=field.columns[1:])
        season_age_info = []
        #        base_info including age and season for all disease
        cause = list(self.cause.cause.drop_duplicates())
        season_age_info = pd.DataFrame([[1] * len(cause)], columns=cause)
        season_age_info.Age_RV = 1 - (age <= 42)
        season_age_info.Age_Ecoli = 1 - ((age <= 35) & (age >= 28))
        season_age_info.Age_clostridiu = 1 - (age <= 35)
        season_age_info.Season_PED = 1 - ((date.month < 5) | (date.month > 11))
        season_age_info.Season_RV = 1 - ((date.month < 4) | (date.month > 10))
        season_age_info.Season_Salm = 1 - ((date.month < 11) & (date.month > 7))
        season_age_info.Season_clostridiu = 1 - ((date.month < 6) & (date.month > 2))
        non_symp = list(self.symptom.symptom.drop_duplicates()[
                            self.symptom.symptom.drop_duplicates().apply(lambda x: x not in symp)])
        symp = pd.DataFrame([[0] * len(symp)], columns=symp)
        non_symp = pd.DataFrame([[1] * len(non_symp)], columns=non_symp)
        obs = pd.concat([season_age_info, symp, non_symp], axis=1)
        if (obs[['腹泻-褐色/黑色/沥青色稀粪', '腹泻-棕/血色稀粪', '腹泻-绿色稀粪', '腹泻-水样稀粪', '粪便带血/血痢', '腹泻-白/灰色稀粪', '腹泻-黄色稀粪']] == 0).sum(
                axis=1).reset_index().rename({0: 'res'}, axis=1).iloc[0, 1] != 0:
            non_symp = list(self.symptom.symptom.drop_duplicates()[self.symptom.symptom.drop_duplicates().apply(
                lambda x: x not in symp and (x not in ['黄色水样', '灰色水样或糊状', '绿色水样', '白色糊状', '暗棕色水样', '红/红棕色/便血']))])
            #             print(non)
            non_symp = pd.DataFrame([[1] * len(non_symp)], columns=non_symp)
            obs = pd.concat([season_age_info, symp, non_symp], axis=1)
        #         print(obs.transpose())
        print('start')
        #         return self.model.predict(obs)

        #         result = self.model.predict_probability(obs)[['PED_0','RV_0','Ecoli_0','Salm_0','clostridiu_C_0','PRRS_0','Mhyo_0','PCV_0','HPS_0','SS_0','Pm_0']]
        result = self.model.predict_probability(obs)
        result.columns = list(self.cause.disease.drop_duplicates())
        result = result.transpose().rename({0: 'disease'}, axis=1)
        result = result[result.disease * 100 >= 0].sort_values('disease', ascending=False).head(5)
        #         result.disease = result.disease.apply(lambda x: str(round(x*100,1))+'%')
        print(result.disease.apply(lambda x: str(round(x * 100, 1)) + '%'))
        self.disease_candidate = list(result.disease.sort_values(ascending=False).head(3).index)
        return result.disease.apply(lambda x: str(round(x * 100, 1)) + '%')

    #     def treat(self, disease_candidate = [],method = []):
    #         if disease_candidate == []:
    #             self.recipe = self.medicine[self.medicine.disease.apply(lambda x: x in self.disease_candidate)]
    #         else:
    #             self.recipe = self.medicine[self.medicine.disease.apply(lambda x: x in disease_candidate)]
    #         if method ==[]:method = self.method
    #         if method == 'group' :self.recipe = self.recipe[self.recipe.method.apply(lambda x: x not in ['粉针','注射液'])]
    #         self.recipe.to_excel('res.xlsx')
    #         print(self.recipe)

    def sample_recom(self):
        res = []
        #         print(self.symp)
        for i in self.symp:
            res.append(self.sample[self.sample.symptom.apply(lambda x: i in x.split(','))][['sample', 'sus']])
        if res == []: return
        res = pd.concat(res).reset_index().drop('index', axis=1)
        res.columns = ['样品采集', '检测项目']
        print(res)
        return res
#         self.recipe.med_ingredient
