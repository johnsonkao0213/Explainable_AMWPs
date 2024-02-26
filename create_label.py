#!/usr/bin/env python
# coding: utf-8

# In[2]:


from src.load_data import *
from src.num_transfer import *
from transformers import BertTokenizer, AdamW, BertTokenizerFast, AutoTokenizer
from transformers import AutoModel, AutoConfig, BertModel, BertConfig, BertForTokenClassification
import torch

import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from collections import Counter


# In[10]:


def check_cnt(l):
    d = Counter(l)
    for i in d.values():
        if i > 1:
            return 1
    return 0

def change_num(num):
    if ',' in num:
        new_num = []
        for c in num:
            if c == ',':
                continue
            new_num.append(c)
        num = ''.join(new_num)
        return str(float(eval(num)))
    else:
        return str(float(eval(num)))

def gen_flag(seg, res):
    reverse_flag = 0
    cnt = 0
    if ';' not in res:
        return reverse_flag
    seq_index = res.index(';')
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
    
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            ss = float(change_num(s[pos.start():pos.end()]))
            if ss in res:
                num_index = res.index(ss)
                cnt+=1
                if num_index > seq_index:
                    reverse_flag = 1
                if cnt == 1:
                    break
    return reverse_flag

def find_substr(res):
    res = list(map(str, res))
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if res[i] in res[j]:
                return 1
    return 0


# In[18]:


def load_alg514_data(filename):  # load the json data to list(dict()) for ALG514 data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    labels = []
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
    reverse_cnt = 0
    label0_cnt, label1_cnt, label2_cnt, label3_cnt = 0, 0, 0, 0
    for d in data:
        id = d['iIndex']
        flag_skip = 0
        if id == 6254 or id == 5652:
            flag_skip = 1
        #if id not in reverse_id:
        #    continue
        #if id != 1075:
        #    continue
        if "lEquations" not in d:
            continue
        x = d['lEquations']
        if len(set(x) - set("0123456789.+-*/()=xXyY; ")) != 0:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['lEquations'] = new_eqs[0]
        else:
            d['lEquations'] = ' ; '.join(new_eqs)
        
        
        #char_index = d['lEquations'].find(';')
        
        if ';' in  d['lEquations']:
            char_index = d['lEquations'].index(';')
        else:
            char_index = 10000000
        seg = d['sQuestion'].strip().split()
        
        #print(char_index)
        #for s in seg:
        #    pos = re.search(pattern, s)
        #    if pos and pos.start() == 0:
        #        ss = change_num(s[pos.start():pos.end()])
            #print(num_first)
        #print(char_index)
        res = []
        def seg_and_tag(st):  # seg the equation and tag the num
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if float(st_num) in res:
                    num_index = d['lEquations'].find(st_num)
                    num_index = d['lEquations'].find(st_num, num_index+1)
                else:
                    num_index = d['lEquations'].find(st_num)
                    
                #print(num_index)
                
                if char_index < num_index and ';' not in res:
                    res.append(';')
                res.append(float(st_num))
                if p_end < len(st):
                    seg_and_tag(st[p_end:])
            return res

        res = seg_and_tag(d['lEquations'])
        #print('==================')
        duplicate = [item for item, count in Counter(res).items() if count > 1]
        #print(duplicate)
        #print(res)
        #if check_cnt(res):
        #    print(id)
        #print("=================================")
        reverse_flag = gen_flag(seg, res)
        #reverse_flag = 0
        if reverse_flag:
            res.reverse()
        #print(reverse_flag)
        
        if ';' in res:
            seq_index = res.index(';')
        else:
            seq_index = 10000000
        new_seg = []
        gt = []
        flag = 0
        for i,s in enumerate(seg):
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                ss = float(change_num(s[pos.start():pos.end()]))
                if ss in res:
                    num_index = res.index(ss)
                    if float(ss) in res and num_index < seq_index and flag==0:
                        flag = 1
                        if duplicate and ss in duplicate and not flag_skip:
                            duplicate.remove(ss)
                            num_index = res.index(ss, num_index+1) #同一num被重複使用
                            if num_index > seq_index:
                                flag = 3
                    elif float(ss) in res and num_index > seq_index and flag==0:
                        flag = 2
                        if duplicate and ss in duplicate and not flag_skip:
                            duplicate.remove(ss)
                            num_index = res.index(ss, num_index+1)
                            if num_index < seq_index:
                                flag = 3
                    elif float(ss) in res and num_index > seq_index and flag==1:
                        flag = 3
                    elif float(ss) in res and num_index < seq_index and flag==2:
                        flag = 3

                    res.remove(ss)
                    #print(res)
                    
                    if ';' in res:
                        seq_index = res.index(';')
                    else:
                        seq_index = 10000000
                    
                    #print(seq_index)
                    #print("=====================")

            #print(flag)
            #print("=====================")
            #print(flag2)
                
            if len(s) == 1 and s in ",.?!;" and flag==1:
                gt.append(1)
                flag = 0
                new_seg.append('[SEP]')
                continue
            elif len(s) == 1 and s in ",.?!;" and flag==2:
                gt.append(2)
                flag = 0
                new_seg.append('[SEP]')
                continue
            elif len(s) == 1 and s in ",.?!;" and flag==3:
                gt.append(3)
                flag = 0
                new_seg.append('[SEP]')
                continue
            elif len(s) == 1 and s in ",.?!;" and flag==0:
                gt.append(0)
                flag = 0
                new_seg.append('[SEP]')
                continue
            new_seg.append(s)
        
        if 3 in gt:
            label3_cnt += 1
            #continue
        if reverse_flag:
            reverse_cnt += 1
            #print(id)
        '''
        if 1 in gt and 2 not in gt:
            print(id)
        if 1 in gt and 0 not in gt:
            print(id)
        '''
        if 2 in gt:
            label2_cnt += 1
        if 1 in gt:
            label1_cnt += 1
        if 0 in gt:
            label0_cnt += 1
        data_dict = {
            "id": id,
            "gt": gt
        }
        #"span_seg": span_seg
        labels.append(data_dict)
        new_seg = new_seg[:-1]
        d['sQuestion'] = ' '.join(new_seg)
        out_data.append(d)
    print(f'reverse count: {reverse_cnt}')
    print(f'label 3 in problem count: {label3_cnt}')
    print(f'label 2 in problem count: {label2_cnt}')
    print(f'label 1 in problem count: {label1_cnt}')
    print(f'label 0 in problem count: {label0_cnt}')
    return out_data, labels


# In[141]:


data_path = "./dataset/alg514/questions_normalization_v5.json"
stage1_path = "./benchmark_labels/label_v3_withQ.json"
data = load_alg514_data(data_path, stage1_path)


# In[21]:


with open('label_v3.json', 'w') as fout:
    json.dump(labels , fout, indent=4)


# In[11]:


def rewrite_alg514_data(filename, output_filename):  # load the json data to list(dict()) for ALG514 data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    labels = []
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
    reverse_cnt = 0
    #label0_cnt, label1_cnt, label2_cnt, label3_cnt = 0, 0, 0, 0
    for d in data:
        id = d['iIndex']
        #flag_skip = 0
        #if id == 6254 or id == 5652:
        #    flag_skip = 1

        if "lEquations" not in d:
            continue
        x = d['lEquations']
        if len(set(x) - set("0123456789.+-*/()=xXyY; ")) != 0:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['lEquations'] = new_eqs[0]
        else:
            d['lEquations'] = ' ; '.join(new_eqs)
        
        
        
        if ';' in  d['lEquations']:
            char_index = d['lEquations'].index(';')
        else:
            char_index = 10000000
        seg = d['sQuestion'].strip().split()
        

        res = []
        def seg_and_tag(st):  # seg the equation and tag the num
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if float(st_num) in res:
                    num_index = d['lEquations'].find(st_num)
                    num_index = d['lEquations'].find(st_num, num_index+1)
                else:
                    num_index = d['lEquations'].find(st_num)
                    

                
                if char_index < num_index and ';' not in res:
                    res.append(';')
                res.append(float(st_num))
                if p_end < len(st):
                    seg_and_tag(st[p_end:])
            return res

        res = seg_and_tag(d['lEquations'])
        reverse_flag = gen_flag(seg, res)
        #reverse_flag = 0
        if reverse_flag:
            reverse_cnt += 1
            #res.reverse()
            x = d['lEquations']
            eqs = x.split(';')
            new_eqs = []
            for eq in eqs:
                sub_eqs = eq.split('=')
                new_sub_eqs = []
                for s_eq in sub_eqs:
                    new_sub_eqs.append(remove_brackets(s_eq.strip()))
                new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
            if len(new_eqs) == 1:
                d['lEquations'] = new_eqs[0]
            else:
                new_eqs.reverse()
                d['lEquations'] = ' ; '.join(new_eqs)
        out_data.append(d)
    
    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)


# In[12]:


rewrite_alg514_data('./dataset/alg514/questions_v2.json', './dataset/alg514/questions_normalization_v2.json')


# In[1]:


from preprocess import *


# In[3]:


preprocess_alg514(data_path, './dataset/alg514/raw/questions.json', './dataset/alg514/questions_v2.json')


# In[37]:


def func(stage1_span_ids_var):
    masks_1 = (stage1_span_ids_var == 1) | (stage1_span_ids_var == 3) | (stage1_span_ids_var == 4)
    masks_2 = (stage1_span_ids_var == 2) | (stage1_span_ids_var == 3) | (stage1_span_ids_var == 4)
    return int(masks_1), int(masks_2)


# In[224]:


def load_alg514_data(filename, stage1_path, output_filename, punctuation=False):  # load the json data to list(dict()) for ALG514 data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    f = open(stage1_path, encoding="utf-8")
    stage1_data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        labels = next(i['gt'] for i in stage1_data if i['id'] == d['iIndex'])
        tag2id = {0: 0, 1: 1, 2: 2, 3: 3, 'Q': 4, 'Q+1':5, 'Q+2':6, 'Q+3':7}
        d['stage1_span'] = [tag2id[i] for i in labels]
        
        
        if "lEquations" not in d:
            continue
        x = d['lEquations']
        if len(set(x) - set("0123456789.+-*/()=xXyY; ")) != 0:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['equation_index'] = 2
            d['lEquations'] = new_eqs[0]
            d['stage1_span_v2'] = [func(tag2id[i])[0] for i in labels]  
            out_data.append(d)
        else:
            #d['lEquations'] = new_eqs[0]
            for i in range(len(new_eqs)):
                d['equation_index'] = i
                d['lEquations'] = new_eqs[i]
                d['stage1_span_v2'] = [func(tag2id[j])[i] for j in labels]
                dd = copy.deepcopy(d)
                out_data.append(dd)
            #continue
            #d['lEquations'] = ' ; '.join(new_eqs)


    
    #return out_data
    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)


# In[4]:


f = open('./dataset/alg514/questions_normalization_v6_v1.json', encoding="utf-8")
data = json.load(f)
f.close()


# In[5]:


tmp = ""
for d in data:
    if d['var_cnt'] == 1:
        d['prev_eq'] = None
    else:
        if d['equation_index'] == 0:
            tmp = d['lEquations']
            d['prev_eq'] = None
        else:
            d['prev_eq'] = tmp


# In[6]:


with open('./dataset/alg514/questions_normalization_v6_v2.json', 'w',encoding="utf-8") as f:
    json.dump(data, f, indent=4)


# In[7]:


data = load_alg514_data('./dataset/alg514/questions_normalization_v6_v2.json', "./benchmark_labels/label_v3_withQ.json")
pairs, generate_nums, copy_nums = transfer_alg514_num(data)


# In[225]:


load_alg514_data('./dataset/alg514/questions_normalization_v3.json', "./benchmark_labels/label_v3_withQ.json", './dataset/alg514/questions_normalization_v6.json', punctuation=False)


# In[6]:


pattern = re.compile("\-?\d+,\d+|\-?\d+\.\d+|\-?\d+|\-?\d+\.\d+%?|\-?\d+%?")
#pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
pairs = []
generate_nums = []
generate_nums_dict = {}
nums = []
copy_nums = 0
input_seq = []
seg = "the difference between 2 times a number and -8 is -12 . find the number .".strip().split()
#equations = d['lEquations']

for s in seg:
    pos = re.search(pattern, s)
    if pos and pos.start() == 0:
        nums.append(s[pos.start():pos.end()])
        input_seq.append('[NUM]')
        if pos.end() < len(s):
            input_seq.append(s[pos.end():])
    else:
        input_seq.append(s)

if copy_nums < len(nums):
    copy_nums = len(nums)

nums_fraction = []

for num in nums:
    if re.search("\d*\(\d+/\d+\)\d*", num):
        nums_fraction.append(num)
nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
# print(nums)
#print(nums_fraction)
float_nums = []
for num in nums:
    if ',' in num:
        new_num = []
        for c in num:
            if c == ',':
                continue
            new_num.append(c)
        num = ''.join(new_num)
        float_nums.append(str(float(eval(num))))
    else:
        float_nums.append(str(float(eval(num))))

float_nums_fraction = []
for num in nums_fraction:
    if ',' in num:
        new_num = []
        for c in num:
            if c == ',':
                continue
            new_num.append(c)
        num = ''.join(new_num)
        float_nums_fraction.append(str(float(eval(num))))
    else:
        float_nums_fraction.append(str(float(eval(num))))
#print(float_nums)
#print(float_nums_fraction)
nums = float_nums
nums_fraction = float_nums_fraction


# In[7]:


def seg_and_tag(st):  # seg the equation and tag the num
    res = []
#     for n in nums_fraction:
#         if n in st:
#             p_start = st.find(n)
#             p_end = p_start + len(n)
#             if p_start > 0:
#                 res += seg_and_tag(st[:p_start])
#             if nums.count(n) == 1:
#                 res.append("N"+str(nums.index(n)))
#             elif nums.count(n) > 1:
#                 # 多个的时候默认使用第一个index代替
#                 res.append("N"+str(nums.index(n)))
#             else:
#                 res.append(n)
#             if p_end < len(st):
#                 res += seg_and_tag(st[p_end:])
#             return res

    #pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    pos_st = re.search("\-?\d+\.\d+%?|\-?\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag(st[:p_start])
        st_num = st[p_start:p_end]
        if nums.count(st_num) == 1:
            res.append("N"+str(nums.index(st_num)))
        elif nums.count(st_num) > 1:
            res.append("N"+str(nums.index(st_num)))
        else:
            res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag(st[p_end:])
        return res
    for ss in st:
        res.append(ss)
    return res


# In[20]:


f = open('./dataset/alg514/questions_normalization_v8.json', encoding="utf-8")
data = json.load(f)
f.close()


# In[21]:


tmp=2598
new_data={}
out_data = []
for d in data:
    if tmp != d['iIndex']:
        out_data.append(new_data)
        new_data={}
    if d['var_cnt'] == 2 and d['equation_index'] == 1:
        new_data['lEquations_2'] = d['lEquations']
        new_data['stage1_span_2'] = d['stage1_span_v2']
        new_data['equation_index_2'] = d['equation_index']
        new_data['prev_eq_2'] = d['prev_eq']
    else:
        new_data['iIndex'] = d['iIndex']
        new_data['sQuestion'] = d['sQuestion']
        new_data['lSolutions'] = d['lSolutions']
        new_data['var_entity'] = d['var_entity']
        new_data['var_cnt'] = d['var_cnt']
        new_data['lEquations_1'] = d['lEquations']
        new_data['stage1_span_1'] = d['stage1_span_v2']
        new_data['equation_index_1'] = d['equation_index']
        new_data['prev_eq_1'] = d['prev_eq']
        tmp = d['iIndex']
        
        if d['equation_index'] == 2:
            new_data['lEquations_2'] = None
            new_data['stage1_span_2'] = None
            new_data['equation_index_2'] = None
            new_data['prev_eq_2'] = None
        


# In[24]:


with open('./dataset/alg514/questions_normalization_v9.json', 'w',encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

