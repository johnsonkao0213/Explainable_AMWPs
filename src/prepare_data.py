import random
import json
import copy
import re
import nltk
from src.data_utils import remove_brackets
from src.lang import InputLang, OutputLang
from src.data_utils import indexes_from_sentence, pad_seq, check_bracket, get_num_stack, get_single_batch_graph, read_json
from itertools import groupby

# def get_train_test_fold(data_path, prefix, data, pairs, group=None):
#     mode_train = 'train'
#     mode_valid = 'valid'
#     mode_test = 'test'
#     train_path = data_path + mode_train + prefix
#     valid_path = data_path + mode_valid + prefix
#     test_path = data_path + mode_test + prefix
#     train = read_json(train_path)
#     train_id = [item['id'] for item in train]
#     valid = read_json(valid_path)
#     valid_id = [item['id'] for item in valid]
#     test = read_json(test_path)
#     test_id = [item['id'] for item in test]
#     train_fold = []
#     valid_fold = []
#     test_fold = []
#
#     for item, pair in zip(data, pairs):
#         pair = list(pair)
#         pair = tuple(pair)
#         if item['id'] in train_id:
#             train_fold.append(pair)
#         elif item['id'] in test_id:
#             test_fold.append(pair)
#         else:
#             valid_fold.append(pair)
#     return train_fold, test_fold, valid_fold

def get_train_test_fold(ori_path,prefix,data,pairs):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item, pair in zip(data, pairs):
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold



def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums,
                 tree=False, use_lm=False, use_group_num=False):
    input_lang = InputLang()
    output_lang = OutputLang()
    train_pairs = []
    test_pairs = []
    print("Indexing words")
    for pair in pairs_trained:
        if len(pair["num_pos"]) > 0:
            if not use_lm:
                input_lang.add_sen_to_vocab(pair["input_seq"])
            output_lang.add_sen_to_vocab(pair["out_seq"])
    if not use_lm:
        input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []  # 用于记录不在输出词典的数字
        for word in pair["out_seq"]:
            temp_num = []
            flag_not = True  # 用检查等式是否存在不在字典的元素
            if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                flag_not = False
                for i, j in enumerate(pair["nums"]): # 遍历nums, 看是否存在
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])  # 生成从0到等式长度的数字

        num_stack.reverse()
        
        if use_lm:
            input_cell = pair["input_seq"]
        else:
            input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)
        if pair["prev_eq"]:
            prev_eq = indexes_from_sentence(output_lang, pair["prev_eq"], tree)
        else:
            prev_eq = []
        train_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "ans": pair['ans'],
            "stage1_span": pair['stage1_span'],
            "stage1_span_len": len(pair['stage1_span']),
            "var_cnt": pair['var_cnt'],
            "var_entity": pair['var_entity'],
            "equation_index": pair['equation_index'],
            "prev_eq": prev_eq,
            "prev_eq_len": len(prev_eq),
            "prev_eq_num_pos": pair['prev_eq_num_pos']
        }

        if use_group_num:
            train_dict['group_num'] = pair['group_num']
#         if pair['id']==1695 or pair['id']==2157:
#             print(train_dict)
        train_pairs.append(train_dict)

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []
        for word in pair["out_seq"]:  # out_seq
            temp_num = []
            flag_not = True
            #print(output_lang.index2word)
            if word not in output_lang.index2word: # 非符号，即word为数字
                flag_not = False
                for i, j in enumerate(pair["nums"]): # nums
                    if j == word:
                        temp_num.append(i) # 在等式的位置信息
            if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                # 数字不在数字列表中，则生成数字列表长度的位置信息，
                # 生成时根据解码器的概率选一个， 参见generate_tree_input
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        
        if use_lm:
            input_cell = pair["input_seq"]
        else:
            input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)
        if pair["prev_eq"]:
            prev_eq = indexes_from_sentence(output_lang, pair["prev_eq"], tree)
        else:
            prev_eq = []
        test_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "ans": pair['ans'],
            "stage1_span": pair['stage1_span'],
            "stage1_span_len": len(pair['stage1_span']),
            "var_cnt": pair['var_cnt'],
            "var_entity": pair['var_entity'], #dict
            "equation_index": pair['equation_index'],
            "target_equation": pair["out_seq"],
            "prev_eq": prev_eq,
            "prev_eq_len": len(prev_eq),
            "prev_eq_num_pos": pair['prev_eq_num_pos']
        }

        if use_group_num:
            test_dict['group_num'] = pair['group_num']
        #if pair['id']==3055 or pair['id']==3591 or pair['id']==6008:
        #    print(test_dict)
        test_pairs.append(test_dict)
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs



def prepare_data_bystep(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums,
                 tree=False, use_lm=False, use_group_num=False):
    input_lang = InputLang()
    output_lang = OutputLang()
    train_pairs = []
    test_pairs = []
    t=0

    print("Indexing words")
    for pair in pairs_trained:
        if len(pair["num_pos"]) > 0:
            if not use_lm:
                input_lang.add_sen_to_vocab(pair["input_seq"])
            output_lang.add_sen_to_vocab(pair["out_seq"])
            if pair["out_seq2"]:
                output_lang.add_sen_to_vocab(pair["out_seq2"])
    if not use_lm:
        input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []  # 用于记录不在输出词典的数字
        for word in pair["out_seq"]:
            temp_num = []
            flag_not = True  # 用检查等式是否存在不在字典的元素
            if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                flag_not = False
                for i, j in enumerate(pair["nums"]): # 遍历nums, 看是否存在
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])  # 生成从0到等式长度的数字

        num_stack.reverse()
        
        num_stack2 = []  # 用于记录不在输出词典的数字
        if pair["out_seq2"]:
            for word in pair["out_seq2"]:
                temp_num = []
                flag_not = True  # 用检查等式是否存在不在字典的元素
                if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                    flag_not = False
                    for i, j in enumerate(pair["nums"]): # 遍历nums, 看是否存在
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack2.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack2.append([_ for _ in range(len(pair["nums"]))])  # 生成从0到等式长度的数字
        num_stack2.reverse()
        
        if use_lm:
            input_cell = pair["input_seq"]
        else:
            input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)
        if pair["out_seq2"]:
            output_cell2 = indexes_from_sentence(output_lang, pair["out_seq2"], tree)
        else:
            output_cell2 = []
#         if pair["prev_eq"]:
#             prev_eq = indexes_from_sentence(output_lang, pair["prev_eq"], tree)
#         else:
#             prev_eq = []
        train_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "output_cell2": output_cell2,
            "output_cell2_len": len(output_cell2),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "num_stack2": num_stack2,
            "ans": pair['ans'],
            "stage1_span": pair['stage1_span'],
            "stage1_span_len": len(pair['stage1_span']),
            "stage1_span2": pair['stage1_span2'],
            "stage1_span2_len": len(pair['stage1_span2']),
            "var_cnt": pair['var_cnt'],
            "var_entity": pair['var_entity'],
            "equation_index": pair['equation_index'],
            "equation_index2": pair['equation_index2'],
        }
#         "prev_eq": prev_eq,
#         "prev_eq_len": len(prev_eq)
#         if len(prev_eq) > t:
#             t=len(prev_eq)
        if use_group_num:
            train_dict['group_num'] = pair['group_num']
#         if pair['id']==1695 or pair['id']==2157:
#             print(train_dict)
        train_pairs.append(train_dict)

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []
        for word in pair["out_seq"]:  # out_seq
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word: # 非符号，即word为数字
                flag_not = False
                for i, j in enumerate(pair["nums"]): # nums
                    if j == word:
                        temp_num.append(i) # 在等式的位置信息
            if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        
        num_stack2 = []
        if pair["out_seq2"]:
            for word in pair["out_seq2"]:  # out_seq
                temp_num = []
                flag_not = True
                #print(output_lang.index2word)
                if word not in output_lang.index2word: # 非符号，即word为数字
                    flag_not = False
                    for i, j in enumerate(pair["nums"]): # nums
                        if j == word:
                            temp_num.append(i) # 在等式的位置信息
                if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                    num_stack2.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack2.append([_ for _ in range(len(pair["nums"]))])
        num_stack2.reverse()
        
        if use_lm:
            input_cell = pair["input_seq"]
        else:
            input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)
        if pair["out_seq2"]:
            output_cell2 = indexes_from_sentence(output_lang, pair["out_seq2"], tree)
        else:
            output_cell2 = []
#         if pair["prev_eq"]:
#             prev_eq = indexes_from_sentence(output_lang, pair["prev_eq"], tree)
#         else:
#             prev_eq = []
        test_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "output_cell2": output_cell2,
            "output_cell2_len": len(output_cell2),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "num_stack2": num_stack2,
            "ans": pair['ans'],
            "stage1_span": pair['stage1_span'],
            "stage1_span_len": len(pair['stage1_span']),
            "stage1_span2": pair['stage1_span2'],
            "stage1_span2_len": len(pair['stage1_span2']),
            "var_cnt": pair['var_cnt'],
            "var_entity": pair['var_entity'], #dict
            "equation_index": pair['equation_index'],
            "equation_index2": pair['equation_index2'],
            "target_equation": pair["out_seq"],
            "target_equation2": pair["out_seq2"],
        }
#                     "prev_eq": prev_eq,
#             "prev_eq_len": len(prev_eq)
        if use_group_num:
            test_dict['group_num'] = pair['group_num']
        #if pair['id']==3055 or pair['id']==3591 or pair['id']==6008:
        #    print(test_dict)
#         if len(prev_eq) > t:
#             t=len(prev_eq)
        test_pairs.append(test_dict)
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


# prepare the batches
# pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len, stage1_span_ids_batches)
def prepare_data_batch(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0,
                        shuffle=True, use_group_num=False, use_lm=False, lm_tokenizer=None):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle:
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    if use_lm:
        attention_mask_batches = []
        token_type_ids_batches = []
        stage1_span_ids_batches = []
        stage1_sentence_ids_batches = []
        attention_mask_sentence_batches = []
        stage1_span_lengths = []
        sentence_length_batches = []
        quantity_indicator_batches = []
        sep_loc_batches = []
        cls_loc_batches = []
        var_cnt_batches = []
        
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []

    if use_group_num:
        group_num_batches = []
        num_graph_batches = []

    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        stage1_span_length = []
        var_cnt_batch = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len)
        for pair in batch:
            if not use_lm:
                input_length.append(pair["input_cell_len"])
            else:
                stage1_span_length.append(pair["stage1_span_len"])
                var_cnt_batch.append(pair["var_cnt"])
                
            output_length.append(pair["output_cell_len"])
                
        # input_lengths.append(input_length)
        output_lengths.append(output_length)
        if not use_lm:
            input_len_max = input_length[0]
        else:
            stage1_len_max = max(stage1_span_length)
            stage1_span_lengths.append(stage1_span_length)
            var_cnt_batches.append(var_cnt_batch)
            
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        if use_lm:
            stage1_span_ids_batch = []
            attention_mask_sentence_batch = []
        if use_group_num:
            group_num_batch = []

        for pair in batch:
            num_batch.append(pair['nums'])
            if use_lm:
                input_batch.append(' '.join(pair['input_cell']))
                stage1_span_padding = pad_seq(pair['stage1_span'], pair['stage1_span_len'], stage1_len_max, pad_token=-100)
                stage1_span_ids_batch.append(stage1_span_padding)
                attention_mask_sentence_batch.append([float(i != -100) for i in stage1_span_padding])
            else:
                input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair["num_stack"])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            if use_group_num and not use_lm:
                group_num_batch.append(pair['group_num'])
            elif use_group_num and use_lm:
                # 要修改
                group_num = pair['group_num']
                input_seq = pair['input_cell']
                new_group_num = []
                pattern = re.compile(r'\[NUM]')

                # update group_num
                acc_count = 0
                temp_input_seq = []
                for idx, s in enumerate(input_seq):
                    if s in ['',  '',  '', '', '', '', '', '']:
                        updated_group_num = []
                        for g_idx in group_num:
                            if g_idx == idx - acc_count:
                                continue
                            elif g_idx > idx - acc_count:
                                updated_group_num.append(g_idx - 1)
                            else:
                                updated_group_num.append(g_idx)
                        acc_count += 1
                        group_num = updated_group_num
                    else:
                        if s != '' and s != '' and s != ' ':
                            temp_input_seq.append(s)
                input_seq = temp_input_seq

                input_seg = []
                seq_mapping = {}
                for idx, s in enumerate(input_seq):
                    pos = re.search(pattern, s)  # 搜索每个词的数字位置
                    if pos and idx in group_num:
                        input_seg.append(s)
                        seq_mapping[idx] = [len(input_seg)-1 + 1]
                    else:
                        seq_mapping[idx] = []
                        # 利用tokenizer来校正group_num
                        lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                        for ss in lm_s:
                            input_seg.append(ss)
                            if idx in group_num:
                                seq_mapping[idx].append(len(input_seg)-1 + 1)
                                # new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                for idx in group_num:
                    if idx < len(input_seq):
                        new_group_num.extend(seq_mapping[idx])

                # for g_idx in group_num:
                #     input_seg = []
                #     for idx, s in enumerate(input_seq):
                #         pos = re.search(pattern, s)  # 搜索每个词的数字位置
                #         if pos and idx in group_num and g_idx == idx:
                #             input_seg.append(s)
                #             new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #         else:
                #             # 利用tokenizer来校正group_num
                #             lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                #             # print(s)
                #             # print(lm_s)
                #             for ss in lm_s:
                #                 input_seg.append(ss)
                #                 if idx in group_num and g_idx == idx:
                #                     new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #             # for ss in s:
                #             #     input_seg.append(ss)
                #             #     if idx in group_num:
                #             #         new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                # check
                # print(pair['id'])
                graph_seq = ""
                for idx in group_num:
                    if idx < len(input_seq):
                        graph_seq += input_seq[idx]

                lm_graph_seq = ""
                lm_seq = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(' '.join(pair['input_cell'])))
                lm_dict = lm_tokenizer(' '.join(pair['input_cell']))
                lm_seq1 = lm_tokenizer.convert_ids_to_tokens(lm_dict['input_ids'])
                for idx in new_group_num:
                    lm_graph_seq += lm_seq[idx].replace("##", '')
                if len(graph_seq.lower()) != len(lm_graph_seq.lower()) - lm_graph_seq.lower().count('[unk]') * 4:
                    print(pair['id'])
                    print(' '.join(pair['input_cell']))
                    print("group_num:", group_num)
                    print(lm_seq)
                    print(lm_seq1)
                    print("new_group_num:", new_group_num)
                    print(graph_seq.lower())
                    print(lm_graph_seq.lower())
                    print(graph_seq.lower() != lm_graph_seq.lower())
                    print(lm_seq1 != lm_seq)
                    print(len(graph_seq.lower()))
                    print(len(lm_graph_seq.lower()) - lm_graph_seq.count('[unk]') * 4)
                    exit(0)

                group_num_batch.append(new_group_num)

        if use_lm:
            input_batch1 = input_batch
            tokens_dict = lm_tokenizer(input_batch, padding=True, add_special_tokens=False)
            input_batch = []  # tokens_dict["input_ids"]
            attention_mask_batch = tokens_dict["attention_mask"]
            token_type_ids_batch = tokens_dict["token_type_ids"]

            num_pos_batch1 = num_pos_batch
            num_pos_batch = []  # need to be updated, so clear it
            quantity_indicator_batch = []
            num_size_batch = []
            sep_loc_batch = []
            cls_loc_batch = []
            #stage1_span_ids_batch1 = stage1_span_ids_batch
            #stage1_span_ids_batch = []
            stage1_sentence_ids_batch = []
            sentence_length_batch = []
            #stage1_span_ids_batch.append(pair['stage1_span'])
            #for input_seq, span_id in zip(tokens_dict["input_ids"], stage1_span_ids_batch1):
            for input_seq in tokens_dict["input_ids"]:
                new_seq = []
                #new_span_id = []
                sentence_length = []
                new_sentence_id = []
                sentence_m = [list(group) for k, group in groupby(input_seq, lambda x: x == 102 or
                                                             x == 0) if not k]
                #for i, j in zip(sentence_m, span_id):
                #    new_span_id.extend([j] * (len(i)+1))
                for sentence_id, i in enumerate(sentence_m):
                    sentence_length.append(len(i)+1)
                    new_sentence_id.extend([sentence_id] * (len(i)+1))
                #print(sentence_length)
                for t_id in input_seq:
                    if t_id == len(lm_tokenizer.vocab):
                        new_seq.append(1)
                    else:
                        new_seq.append(t_id)
                input_batch.append(new_seq)
                #new_span_id.extend([0] * (input_seq.count(0)))
                new_sentence_id.extend([0] * (input_seq.count(0)))
                #stage1_span_ids_batch.append(new_span_id)
                sentence_length_batch.append(sentence_length)
                stage1_sentence_ids_batch.append(new_sentence_id)
                assert all(len(i) == len(j) for i,j in zip(stage1_sentence_ids_batch, input_batch)) 

            for t_idx, input_seq in enumerate(input_batch):
                num_pos = []
                sep_loc = []
                cls_loc = []
                quantity_indicator = []
                for idx, t_id in enumerate(input_seq):
                    # if t_id == lm_tokenizer.vocab['[NUM]']:
                    if t_id == 1:
                        num_pos.append(idx)
                        quantity_indicator.append(1)
                    elif t_id == 102:
                        quantity_indicator.append(0)
                        sep_loc.append(idx)
                    elif t_id == 101:
                        quantity_indicator.append(0)
                        cls_loc.append(idx)
                    else:
                        quantity_indicator.append(0)
                # if len(num_pos) != len(num_pos_batch1[t_idx]):  # 检查一致性
                #     print(id_batch[t_idx])
                #     print(input_batch1[t_idx])
                #     print(lm_tokenizer.convert_ids_to_tokens(input_batch[t_idx]))
                #     print(len(num_pos))
                #     print(len(num_pos_batch1[t_idx]))
                #     print(group_num_batch[t_idx])
                #     exit(0)
                sep_loc_padding = pad_seq(sep_loc, len(sep_loc), stage1_len_max, pad_token=-1)
                cls_loc_padding = pad_seq(cls_loc, len(cls_loc), stage1_len_max, pad_token=-1)
                sep_loc_batch.append(sep_loc_padding)
                cls_loc_batch.append(cls_loc_padding)
                num_pos_batch.append(num_pos)
                quantity_indicator_batch.append(quantity_indicator)
                num_size_batch.append(len(num_pos))
                #input_length.append(input_seq.index(lm_tokenizer.vocab['[SEP]'])+1)
                input_length.append(len(input_seq) - input_seq[::-1].index(lm_tokenizer.vocab['[SEP]']))
            attention_mask_batches.append(attention_mask_batch)
            token_type_ids_batches.append(token_type_ids_batch)
            stage1_sentence_ids_batches.append(stage1_sentence_ids_batch)
            stage1_span_ids_batches.append(stage1_span_ids_batch)
            attention_mask_sentence_batches.append(attention_mask_sentence_batch)
            sentence_length_batches.append(sentence_length_batch)
            quantity_indicator_batches.append(quantity_indicator_batch)
            sep_loc_batches.append(sep_loc_batch)
            cls_loc_batches.append(cls_loc_batch)
            
            

        input_batches.append(input_batch)
        input_lengths.append(input_length)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)

        if use_group_num:
            group_num_batches.append(group_num_batch)
            num_graph_batches.append(get_single_batch_graph(input_batch, input_length, group_num_batch, num_batch, num_pos_batch))

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }

    if use_group_num:
        batches_dict['group_num_batches'] = group_num_batches
        batches_dict['num_graph_batches'] = num_graph_batches

    if use_lm:
        batches_dict['attention_mask_batches'] = attention_mask_batches
        batches_dict['token_type_ids_batches'] = token_type_ids_batches
        batches_dict['stage1_span_ids_batches'] = stage1_span_ids_batches
        batches_dict['stage1_sentence_ids_batches'] = stage1_sentence_ids_batches
        batches_dict['attention_mask_sentence_batches'] = attention_mask_sentence_batches
        batches_dict['stage1_span_lengths'] = stage1_span_lengths
        batches_dict['sentence_length_batches'] = sentence_length_batches
        batches_dict['quantity_indicator_batches'] = quantity_indicator_batches
        batches_dict['sep_loc_batches'] = sep_loc_batches
        batches_dict['cls_loc_batches'] = cls_loc_batches
        batches_dict['var_cnt_batches'] = var_cnt_batches

    return batches_dict


def prepare_data_batch_nosep(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0,
                        shuffle=True, use_group_num=False, use_lm=False, lm_tokenizer=None):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle:
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    if use_lm:
        attention_mask_batches = []
        token_type_ids_batches = []
        stage1_span_ids_batches = []
        stage1_sentence_ids_batches = []
        attention_mask_sentence_batches = []
        stage1_span_lengths = []
        sentence_length_batches = []
        quantity_indicator_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []

    if use_group_num:
        group_num_batches = []
        num_graph_batches = []

    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        stage1_span_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len)
        for pair in batch:
            if not use_lm:
                input_length.append(pair["input_cell_len"])
            else:
                stage1_span_length.append(pair["stage1_span_len"])
            output_length.append(pair["output_cell_len"])
                
        # input_lengths.append(input_length)
        output_lengths.append(output_length)
        if not use_lm:
            input_len_max = input_length[0]
        else:
            stage1_len_max = max(stage1_span_length)
            #print(stage1_len_max)
            stage1_span_lengths.append(stage1_span_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        if use_lm:
            stage1_span_ids_batch = []
            attention_mask_sentence_batch = []
        if use_group_num:
            group_num_batch = []

        for pair in batch:
            num_batch.append(pair['nums'])
            if use_lm:
                input_batch.append(' '.join(pair['input_cell']))
                stage1_span_padding = pad_seq(pair['stage1_span'], pair['stage1_span_len'], stage1_len_max, pad_token=-100)
                stage1_span_ids_batch.append(stage1_span_padding)
                attention_mask_sentence_batch.append([float(i != -100) for i in stage1_span_padding])
            else:
                input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair["num_stack"])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            if use_group_num and not use_lm:
                group_num_batch.append(pair['group_num'])
            elif use_group_num and use_lm:
                # 要修改
                group_num = pair['group_num']
                input_seq = pair['input_cell']
                new_group_num = []
                pattern = re.compile(r'\[NUM]')

                # update group_num
                acc_count = 0
                temp_input_seq = []
                for idx, s in enumerate(input_seq):
                    if s in ['',  '',  '', '', '', '', '', '']:
                        updated_group_num = []
                        for g_idx in group_num:
                            if g_idx == idx - acc_count:
                                continue
                            elif g_idx > idx - acc_count:
                                updated_group_num.append(g_idx - 1)
                            else:
                                updated_group_num.append(g_idx)
                        acc_count += 1
                        group_num = updated_group_num
                    else:
                        if s != '' and s != '' and s != ' ':
                            temp_input_seq.append(s)
                input_seq = temp_input_seq

                input_seg = []
                seq_mapping = {}
                for idx, s in enumerate(input_seq):
                    pos = re.search(pattern, s)  # 搜索每个词的数字位置
                    if pos and idx in group_num:
                        input_seg.append(s)
                        seq_mapping[idx] = [len(input_seg)-1 + 1]
                    else:
                        seq_mapping[idx] = []
                        # 利用tokenizer来校正group_num
                        lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                        for ss in lm_s:
                            input_seg.append(ss)
                            if idx in group_num:
                                seq_mapping[idx].append(len(input_seg)-1 + 1)
                                # new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                for idx in group_num:
                    if idx < len(input_seq):
                        new_group_num.extend(seq_mapping[idx])

                # for g_idx in group_num:
                #     input_seg = []
                #     for idx, s in enumerate(input_seq):
                #         pos = re.search(pattern, s)  # 搜索每个词的数字位置
                #         if pos and idx in group_num and g_idx == idx:
                #             input_seg.append(s)
                #             new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #         else:
                #             # 利用tokenizer来校正group_num
                #             lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                #             # print(s)
                #             # print(lm_s)
                #             for ss in lm_s:
                #                 input_seg.append(ss)
                #                 if idx in group_num and g_idx == idx:
                #                     new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #             # for ss in s:
                #             #     input_seg.append(ss)
                #             #     if idx in group_num:
                #             #         new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                # check
                # print(pair['id'])
                graph_seq = ""
                for idx in group_num:
                    if idx < len(input_seq):
                        graph_seq += input_seq[idx]

                lm_graph_seq = ""
                lm_seq = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(' '.join(pair['input_cell'])))
                lm_dict = lm_tokenizer(' '.join(pair['input_cell']))
                lm_seq1 = lm_tokenizer.convert_ids_to_tokens(lm_dict['input_ids'])
                for idx in new_group_num:
                    lm_graph_seq += lm_seq[idx].replace("##", '')
                if len(graph_seq.lower()) != len(lm_graph_seq.lower()) - lm_graph_seq.lower().count('[unk]') * 4:
                    print(pair['id'])
                    print(' '.join(pair['input_cell']))
                    print("group_num:", group_num)
                    print(lm_seq)
                    print(lm_seq1)
                    print("new_group_num:", new_group_num)
                    print(graph_seq.lower())
                    print(lm_graph_seq.lower())
                    print(graph_seq.lower() != lm_graph_seq.lower())
                    print(lm_seq1 != lm_seq)
                    print(len(graph_seq.lower()))
                    print(len(lm_graph_seq.lower()) - lm_graph_seq.count('[unk]') * 4)
                    exit(0)

                group_num_batch.append(new_group_num)

        if use_lm:
            input_batch1 = input_batch
            tokens_dict = lm_tokenizer(input_batch, padding=True, add_special_tokens=False)
            input_batch = []  # tokens_dict["input_ids"]
            attention_mask_batch = [] # tokens_dict["attention_mask"]
            token_type_ids_batch = [] # tokens_dict["token_type_ids"]
            #attention_mask_batch = tokens_dict["attention_mask"]
            #token_type_ids_batch = tokens_dict["token_type_ids"]
            quantity_indicator_batch = []
            num_pos_batch1 = num_pos_batch
            num_pos_batch = []  # need to be updated, so clear it
            num_size_batch = []
            stage1_sentence_ids_batch = []
            sentence_length_batch = []
            
            for input_seq, mask, type_ids in zip(tokens_dict["input_ids"], tokens_dict["attention_mask"], tokens_dict["token_type_ids"]):
                new_seq = []
                new_mask = []
                new_type_ids = []
                sentence_length = []
                new_sentence_id = []
                sentence_m = [list(group) for k, group in groupby(input_seq, lambda x: x == 102 or
                                                             x == 0) if not k]
                #nosep
                for sentence_id, i in enumerate(sentence_m):
                    sentence_length.append(len(i))
                    new_sentence_id.extend([sentence_id] * len(i))
                
                sentence_length[-1] += 1
                new_sentence_id.extend([len(sentence_m)-1])
                sep_count = input_seq.count(102)
                cnt = 1
                for t_id, t_mask, t_type_ids in zip(input_seq, mask, type_ids):
                    if t_id == len(lm_tokenizer.vocab):
                        new_seq.append(1)
                        new_mask.append(t_mask)
                        new_type_ids.append(t_type_ids)
                    elif t_id == 102 and cnt < sep_count:
                        cnt += 1
                        continue
                    else:
                        new_mask.append(t_mask)
                        new_type_ids.append(t_type_ids)
                        new_seq.append(t_id)
                
                
                
#                 for t_id in input_seq:
#                     if t_id == len(lm_tokenizer.vocab):
#                         new_seq.append(1)
#                     else:
#                         new_seq.append(t_id)
                #new_seq1 = new_seq
                #print(new_seq1)
                new_seq.extend([0] * (sep_count-1))
                new_mask.extend([0] * (sep_count-1))
                new_type_ids.extend([0] * (sep_count-1))
                
                input_batch.append(new_seq)
                attention_mask_batch.append(new_mask)
                token_type_ids_batch.append(new_type_ids)
                #new_span_id.extend([0] * (input_seq.count(0)))
                new_sentence_id.extend([0] * (new_seq.count(0)))
                #stage1_span_ids_batch.append(new_span_id)
                #print(len(new_seq) - new_seq[::-1].index(lm_tokenizer.vocab['[SEP]']))
                #print(sentence_length)
                sentence_length_batch.append(sentence_length)
                stage1_sentence_ids_batch.append(new_sentence_id)
                assert all(len(i) == len(j) for i,j in zip(stage1_sentence_ids_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(attention_mask_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(token_type_ids_batch, input_batch))

            for t_idx, input_seq in enumerate(input_batch):
                num_pos = []
                quantity_indicator = []
                for idx, t_id in enumerate(input_seq):
                    # if t_id == lm_tokenizer.vocab['[NUM]']:
                    if t_id == 1:
                        quantity_indicator.append(1)
                        num_pos.append(idx)
                    else:
                        quantity_indicator.append(0)
                # if len(num_pos) != len(num_pos_batch1[t_idx]):  # 检查一致性
                #     print(id_batch[t_idx])
                #     print(input_batch1[t_idx])
                #     print(lm_tokenizer.convert_ids_to_tokens(input_batch[t_idx]))
                #     print(len(num_pos))
                #     print(len(num_pos_batch1[t_idx]))
                #     print(group_num_batch[t_idx])
                #     exit(0)

                num_pos_batch.append(num_pos)
                quantity_indicator_batch.append(quantity_indicator)
                num_size_batch.append(len(num_pos))
                #input_length.append(input_seq.index(lm_tokenizer.vocab['[SEP]'])+1)
                input_length.append(len(input_seq) - input_seq[::-1].index(lm_tokenizer.vocab['[SEP]']))
            attention_mask_batches.append(attention_mask_batch)
            token_type_ids_batches.append(token_type_ids_batch)
            stage1_sentence_ids_batches.append(stage1_sentence_ids_batch)
            stage1_span_ids_batches.append(stage1_span_ids_batch)
            attention_mask_sentence_batches.append(attention_mask_sentence_batch)
            sentence_length_batches.append(sentence_length_batch)
            quantity_indicator_batches.append(quantity_indicator_batch)
            #print(input_length)
            

        input_batches.append(input_batch)
        input_lengths.append(input_length)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)

        if use_group_num:
            group_num_batches.append(group_num_batch)
            num_graph_batches.append(get_single_batch_graph(input_batch, input_length, group_num_batch, num_batch, num_pos_batch))

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }

    if use_group_num:
        batches_dict['group_num_batches'] = group_num_batches
        batches_dict['num_graph_batches'] = num_graph_batches

    if use_lm:
        batches_dict['attention_mask_batches'] = attention_mask_batches
        batches_dict['token_type_ids_batches'] = token_type_ids_batches
        batches_dict['stage1_span_ids_batches'] = stage1_span_ids_batches
        batches_dict['stage1_sentence_ids_batches'] = stage1_sentence_ids_batches
        batches_dict['attention_mask_sentence_batches'] = attention_mask_sentence_batches
        batches_dict['stage1_span_lengths'] = stage1_span_lengths
        batches_dict['sentence_length_batches'] = sentence_length_batches
        batches_dict['quantity_indicator_batches'] = quantity_indicator_batches

    return batches_dict

# prepare the batches
# pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len, stage1_span_ids_batches)
def prepare_data_batch_origin(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0,
                        shuffle=True, use_group_num=False, use_lm=False, lm_tokenizer=None):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle:
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    if use_lm:
        attention_mask_batches = []
        token_type_ids_batches = []
        stage1_span_ids_batches = []
        stage1_sentence_ids_batches = []
        quantity_indicator_batches = []
        cls_loc_batches = []
        attention_mask_sentence_batches = []
        var_cnt_batches = []
        equation_id_batches = []
        prev_eq_batches = []
        prev_eq_len_batches = []
        prev_eq_num_pos_batches = []
        #attention_mask_prev_eq_batches = []
        #stage1_span_lengths = []
        #sentence_length_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []

    if use_group_num:
        group_num_batches = []
        num_graph_batches = []

    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        #stage1_span_length = []
        var_cnt_batch = []
        equation_id_batch = []
        prev_eq_len_batch = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len)
        for pair in batch:
            if not use_lm:
                input_length.append(pair["input_cell_len"])
            else:
                #stage1_span_length.append(pair["stage1_span_len"])
                if pair["prev_eq_len"] == 0:
                    prev_eq_len_batch.append(pair["prev_eq_len"]+1)
                else:
                    prev_eq_len_batch.append(pair["prev_eq_len"])
                var_cnt_batch.append(pair["var_cnt"])
                equation_id_batch.append(pair["equation_index"])
                
            output_length.append(pair["output_cell_len"])
                
        # input_lengths.append(input_length)
        output_lengths.append(output_length)
        if not use_lm:
            input_len_max = input_length[0]
        else:
            stage1_len_max = 9
            var_cnt_batches.append(var_cnt_batch)
            equation_id_batches.append(equation_id_batch)
            prev_eq_max = max(prev_eq_len_batch)
            prev_eq_len_batches.append(prev_eq_len_batch)
#             stage1_span_lengths.append(stage1_span_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        if use_lm:
            stage1_span_ids_batch = []
            prev_eq_batch = []
            prev_eq_num_pos_batch = []
            #attention_mask_prev_eq_batch = []
            #attention_mask_sentence_batch = []
        if use_group_num:
            group_num_batch = []
        if prev_eq_max == 0:
            prev_eq_max = 1
        for pair in batch:
            num_batch.append(pair['nums'])
            if use_lm:
                input_batch.append(' '.join(pair['input_cell']))
                stage1_span_ids_batch.append(pair['stage1_span'])
                prev_eq_batch.append(pad_seq(pair['prev_eq'], pair['prev_eq_len'], prev_eq_max, pad_token=outlang_pad_token))
                prev_eq_num_pos_batch.append(pair['prev_eq_num_pos'])
                #attention_mask_prev_eq_batch.append([float(i != 0) for i in prev_eq_batch])
                #stage1_span_padding = pad_seq(pair['stage1_span'], pair['stage1_span_len'], stage1_len_max, pad_token=-100)
                #stage1_span_ids_batch.append(stage1_span_padding)
                #attention_mask_sentence_batch.append([float(i != -100) for i in stage1_span_padding])
            else:
                input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair["num_stack"])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            if use_group_num and not use_lm:
                group_num_batch.append(pair['group_num'])
            elif use_group_num and use_lm:
                # 要修改
                group_num = pair['group_num']
                input_seq = pair['input_cell']
                new_group_num = []
                pattern = re.compile(r'\[NUM]')

                # update group_num
                acc_count = 0
                temp_input_seq = []
                for idx, s in enumerate(input_seq):
                    if s in ['',  '',  '', '', '', '', '', '']:
                        updated_group_num = []
                        for g_idx in group_num:
                            if g_idx == idx - acc_count:
                                continue
                            elif g_idx > idx - acc_count:
                                updated_group_num.append(g_idx - 1)
                            else:
                                updated_group_num.append(g_idx)
                        acc_count += 1
                        group_num = updated_group_num
                    else:
                        if s != '' and s != '' and s != ' ':
                            temp_input_seq.append(s)
                input_seq = temp_input_seq

                input_seg = []
                seq_mapping = {}
                for idx, s in enumerate(input_seq):
                    pos = re.search(pattern, s)  # 搜索每个词的数字位置
                    if pos and idx in group_num:
                        input_seg.append(s)
                        seq_mapping[idx] = [len(input_seg)-1 + 1]
                    else:
                        seq_mapping[idx] = []
                        # 利用tokenizer来校正group_num
                        lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                        for ss in lm_s:
                            input_seg.append(ss)
                            if idx in group_num:
                                seq_mapping[idx].append(len(input_seg)-1 + 1)
                                # new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                for idx in group_num:
                    if idx < len(input_seq):
                        new_group_num.extend(seq_mapping[idx])

                # for g_idx in group_num:
                #     input_seg = []
                #     for idx, s in enumerate(input_seq):
                #         pos = re.search(pattern, s)  # 搜索每个词的数字位置
                #         if pos and idx in group_num and g_idx == idx:
                #             input_seg.append(s)
                #             new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #         else:
                #             # 利用tokenizer来校正group_num
                #             lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                #             # print(s)
                #             # print(lm_s)
                #             for ss in lm_s:
                #                 input_seg.append(ss)
                #                 if idx in group_num and g_idx == idx:
                #                     new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #             # for ss in s:
                #             #     input_seg.append(ss)
                #             #     if idx in group_num:
                #             #         new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                # check
                # print(pair['id'])
                graph_seq = ""
                for idx in group_num:
                    if idx < len(input_seq):
                        graph_seq += input_seq[idx]

                lm_graph_seq = ""
                lm_seq = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(' '.join(pair['input_cell'])))
                lm_dict = lm_tokenizer(' '.join(pair['input_cell']))
                lm_seq1 = lm_tokenizer.convert_ids_to_tokens(lm_dict['input_ids'])
                for idx in new_group_num:
                    lm_graph_seq += lm_seq[idx].replace("##", '')
                if len(graph_seq.lower()) != len(lm_graph_seq.lower()) - lm_graph_seq.lower().count('[unk]') * 4:
                    print(pair['id'])
                    print(' '.join(pair['input_cell']))
                    print("group_num:", group_num)
                    print(lm_seq)
                    print(lm_seq1)
                    print("new_group_num:", new_group_num)
                    print(graph_seq.lower())
                    print(lm_graph_seq.lower())
                    print(graph_seq.lower() != lm_graph_seq.lower())
                    print(lm_seq1 != lm_seq)
                    print(len(graph_seq.lower()))
                    print(len(lm_graph_seq.lower()) - lm_graph_seq.count('[unk]') * 4)
                    exit(0)

                group_num_batch.append(new_group_num)

        if use_lm:
            input_batch1 = input_batch
            tokens_dict = lm_tokenizer(input_batch, padding=True, add_special_tokens=False)
            input_batch = []  # tokens_dict["input_ids"]
            #attention_mask_batch = [] # tokens_dict["attention_mask"]
            #token_type_ids_batch = [] # tokens_dict["token_type_ids"]
            attention_mask_batch = tokens_dict["attention_mask"]
            token_type_ids_batch = tokens_dict["token_type_ids"]

            num_pos_batch1 = num_pos_batch
            num_pos_batch = []  # need to be updated, so clear it
            num_size_batch = []
            stage1_span_ids_batch1 = stage1_span_ids_batch
            stage1_span_ids_batch = []
            quantity_indicator_batch = []
            stage1_sentence_ids_batch = []
            cls_loc_batch = []
            attention_mask_sentence_batch = []
            prev_eq_num_pos_batch1 = prev_eq_num_pos_batch
            prev_eq_num_pos_batch = []
            #stage1_span_ids_batch.append(pair['stage1_span'])
            #for input_seq, span_id, mask, type_ids in zip(tokens_dict["input_ids"], stage1_span_ids_batch1, attention_mask_batch, token_type_ids_batch):
            for input_seq, span_id in zip(tokens_dict["input_ids"], stage1_span_ids_batch1):
            #for input_seq in tokens_dict["input_ids"]:
                new_seq = []
                new_span_id = []
                new_sentence_id = []
                sentence_m = [list(group) for k, group in groupby(input_seq, lambda x: x == 102 or
                                                             x == 0) if not k]
                for i, j in zip(sentence_m, span_id):
                    new_span_id.extend([j] * (len(i)+1))
                
                #For no [sep] token
                #for i, j in zip(sentence_m, span_id):
                #    new_span_id.extend([j] * len(i))
                
                
                #for idx, t_id in enumerate(input_seq):
                #    if t_id == len(lm_tokenizer.vocab):
                #        new_seq.append(1)
                #    elif t_id == 102:
                #        mask.pop(idx)
                #        type_ids.pop(idx)
                #        continue
                #    else:
                #        new_seq.append(t_id)
                #attention_mask_batch.append(mask)
                #token_type_ids_batch.append(type_ids)
                for sentence_id, i in enumerate(sentence_m):
                    new_sentence_id.extend([sentence_id] * (len(i)+1))
                
                for t_id in input_seq:
                    if t_id == len(lm_tokenizer.vocab):
                        new_seq.append(1)
                    else:
                        new_seq.append(t_id)
                input_batch.append(new_seq)
                new_span_id.extend([0] * (input_seq.count(0)))
                new_sentence_id.extend([0] * (input_seq.count(0)))
                stage1_span_ids_batch.append(new_span_id)
                stage1_sentence_ids_batch.append(new_sentence_id)
                assert all(len(i) == len(j) for i,j in zip(stage1_span_ids_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(stage1_sentence_ids_batch, input_batch))

            for t_idx, input_seq in enumerate(input_batch):
                num_pos = []
                cls_loc = []
                quantity_indicator = []
                for idx, t_id in enumerate(input_seq):
                    # if t_id == lm_tokenizer.vocab['[NUM]']:
                    if t_id == 1:
                        num_pos.append(idx)
                        #quantity_indicator.append(1)
                        if stage1_span_ids_batch[t_idx][idx] == 1:
                            quantity_indicator.append(1)
                        else:
                            quantity_indicator.append(0)
                    elif t_id == 101 and stage1_span_ids_batch[t_idx][idx] == 1:
                    #elif t_id == 101:
                        cls_loc.append(idx)
                        quantity_indicator.append(0)
                    else:
                        quantity_indicator.append(0)
                # if len(num_pos) != len(num_pos_batch1[t_idx]):  # 检查一致性
                #     print(id_batch[t_idx])
                #     print(input_batch1[t_idx])
                #     print(lm_tokenizer.convert_ids_to_tokens(input_batch[t_idx]))
                #     print(len(num_pos))
                #     print(len(num_pos_batch1[t_idx]))
                #     print(group_num_batch[t_idx])
                #     exit(0)
                #prev_eq_num_pos = []
                prev_eq_num_pos = [0] * len(input_seq)
                for i in prev_eq_num_pos_batch1[t_idx]:
                    #prev_eq_num_pos.append(num_pos[i])
                    prev_eq_num_pos[num_pos[i]] = 1
                prev_eq_num_pos_batch.append(prev_eq_num_pos)
                
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                quantity_indicator_batch.append(quantity_indicator)
                cls_loc_padding = pad_seq(cls_loc, len(cls_loc), stage1_len_max, pad_token=-1)
                attention_mask_sentence_batch.append([float(i != -1) for i in cls_loc_padding])
                cls_loc_batch.append(cls_loc_padding)
                #input_length.append(input_seq.index(lm_tokenizer.vocab['[SEP]'])+1)
                input_length.append(len(input_seq) - input_seq[::-1].index(lm_tokenizer.vocab['[SEP]']))
            attention_mask_batches.append(attention_mask_batch)
            token_type_ids_batches.append(token_type_ids_batch)
            stage1_sentence_ids_batches.append(stage1_sentence_ids_batch)
            stage1_span_ids_batches.append(stage1_span_ids_batch)
            attention_mask_sentence_batches.append(attention_mask_sentence_batch)
            #sentence_length_batches.append(sentence_length_batch)
            quantity_indicator_batches.append(quantity_indicator_batch)
            cls_loc_batches.append(cls_loc_batch)
            prev_eq_batches.append(prev_eq_batch)
            prev_eq_num_pos_batches.append(prev_eq_num_pos_batch)
            #attention_mask_prev_eq_batches.append(attention_mask_prev_eq_batch)
            
        
        input_batches.append(input_batch)
        input_lengths.append(input_length)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)

        if use_group_num:
            group_num_batches.append(group_num_batch)
            num_graph_batches.append(get_single_batch_graph(input_batch, input_length, group_num_batch, num_batch, num_pos_batch))

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }

    if use_group_num:
        batches_dict['group_num_batches'] = group_num_batches
        batches_dict['num_graph_batches'] = num_graph_batches

    if use_lm:
        batches_dict['attention_mask_batches'] = attention_mask_batches
        batches_dict['token_type_ids_batches'] = token_type_ids_batches
        batches_dict['stage1_span_ids_batches'] = stage1_span_ids_batches
        batches_dict['stage1_sentence_ids_batches'] = stage1_sentence_ids_batches
        batches_dict['quantity_indicator_batches'] = quantity_indicator_batches
        batches_dict['attention_mask_sentence_batches'] = attention_mask_sentence_batches
        batches_dict['cls_loc_batches'] = cls_loc_batches
        batches_dict['var_cnt_batches'] = var_cnt_batches
        batches_dict['equation_id_batches'] = equation_id_batches
        batches_dict['prev_eq_batches'] = prev_eq_batches
        batches_dict['prev_eq_len_batches'] = prev_eq_len_batches
        batches_dict['prev_eq_num_pos_batches'] = prev_eq_num_pos_batches
        #batches_dict['attention_mask_prev_eq_batches'] = attention_mask_prev_eq_batches
        #batches_dict['stage1_span_lengths'] = stage1_span_lengths

    return batches_dict



def prepare_data_batch_origin_bystep(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0,
                        shuffle=True, use_group_num=False, use_lm=False, lm_tokenizer=None):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle:
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_lengths = []
    output_lengths = []
    output2_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    if use_lm:
        attention_mask_batches = []
        token_type_ids_batches = []
        stage1_span_ids_batches = []
        stage1_span_ids2_batches = []
        stage1_sentence_ids_batches = []
        quantity_indicator_batches = []
        cls_loc_batches = []
        cls_loc2_batches = []
        attention_mask_sentence_batches = []
        attention_mask_sentence2_batches = []
        var_cnt_batches = []
        equation_id_batches = []
        equation_id2_batches = []
        #stage1_span_lengths = []
        #sentence_length_batches = []
    output_batches = []
    output2_batches = []
    num_stack_batches = []  # save the num stack which
    num_stack2_batches = []
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []

    if use_group_num:
        group_num_batches = []
        num_graph_batches = []

    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        output2_length = []
        #stage1_span_length = []
        var_cnt_batch = []
        equation_id_batch = []
        equation_id2_batch = []
        #prev_eq_len_batch = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len)
        for pair in batch:
            if not use_lm:
                input_length.append(pair["input_cell_len"])
            else:
                #stage1_span_length.append(pair["stage1_span_len"])
                #prev_eq_len_batch.append(pair["prev_eq_len"])
                var_cnt_batch.append(pair["var_cnt"])
                equation_id_batch.append(pair["equation_index"])
                equation_id2_batch.append(pair["equation_index2"])
                
            output_length.append(pair["output_cell_len"])
            output2_length.append(pair["output_cell2_len"])
                
        # input_lengths.append(input_length)
        output_lengths.append(output_length)
        output2_lengths.append(output2_length)
        if not use_lm:
            input_len_max = input_length[0]
        else:
            stage1_len_max = 9
            var_cnt_batches.append(var_cnt_batch)
            equation_id_batches.append(equation_id_batch)
            equation_id2_batches.append(equation_id2_batch)
#             prev_eq_max = max(prev_eq_len_batch)
#             prev_eq_len_batches.append(prev_eq_len_batch)
#             stage1_span_lengths.append(stage1_span_length)
        output_len_max = max(output_length)
        output2_len_max = max(output2_length)
        input_batch = []
        output_batch = []
        output2_batch = []
        num_batch = []
        num_stack_batch = []
        num_stack2_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        if use_lm:
            stage1_span_ids_batch = []
            stage1_span_ids2_batch = []
            #prev_eq_batch = []
            #attention_mask_prev_eq_batch = []
            #attention_mask_sentence_batch = []
        if use_group_num:
            group_num_batch = []

        for pair in batch:
            num_batch.append(pair['nums'])
            if use_lm:
                input_batch.append(' '.join(pair['input_cell']))
                stage1_span_ids_batch.append(pair['stage1_span'])
                stage1_span_ids2_batch.append(pair['stage1_span2'])
                #prev_eq_batch.append(pad_seq(pair['prev_eq'], pair['prev_eq_len'], prev_eq_max, pad_token=outlang_pad_token))
                #attention_mask_prev_eq_batch.append([float(i != 0) for i in prev_eq_batch])
                #stage1_span_padding = pad_seq(pair['stage1_span'], pair['stage1_span_len'], stage1_len_max, pad_token=-100)
                #stage1_span_ids_batch.append(stage1_span_padding)
                #attention_mask_sentence_batch.append([float(i != -100) for i in stage1_span_padding])
            else:
                input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            output2_batch.append(pad_seq(pair['output_cell2'], pair['output_cell2_len'], output2_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair["num_stack"])
            num_stack2_batch.append(pair["num_stack2"])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            if use_group_num and not use_lm:
                group_num_batch.append(pair['group_num'])
            elif use_group_num and use_lm:
                # 要修改
                group_num = pair['group_num']
                input_seq = pair['input_cell']
                new_group_num = []
                pattern = re.compile(r'\[NUM]')

                # update group_num
                acc_count = 0
                temp_input_seq = []
                for idx, s in enumerate(input_seq):
                    if s in ['',  '',  '', '', '', '', '', '']:
                        updated_group_num = []
                        for g_idx in group_num:
                            if g_idx == idx - acc_count:
                                continue
                            elif g_idx > idx - acc_count:
                                updated_group_num.append(g_idx - 1)
                            else:
                                updated_group_num.append(g_idx)
                        acc_count += 1
                        group_num = updated_group_num
                    else:
                        if s != '' and s != '' and s != ' ':
                            temp_input_seq.append(s)
                input_seq = temp_input_seq

                input_seg = []
                seq_mapping = {}
                for idx, s in enumerate(input_seq):
                    pos = re.search(pattern, s)  # 搜索每个词的数字位置
                    if pos and idx in group_num:
                        input_seg.append(s)
                        seq_mapping[idx] = [len(input_seg)-1 + 1]
                    else:
                        seq_mapping[idx] = []
                        # 利用tokenizer来校正group_num
                        lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                        for ss in lm_s:
                            input_seg.append(ss)
                            if idx in group_num:
                                seq_mapping[idx].append(len(input_seg)-1 + 1)
                                # new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                for idx in group_num:
                    if idx < len(input_seq):
                        new_group_num.extend(seq_mapping[idx])

                # for g_idx in group_num:
                #     input_seg = []
                #     for idx, s in enumerate(input_seq):
                #         pos = re.search(pattern, s)  # 搜索每个词的数字位置
                #         if pos and idx in group_num and g_idx == idx:
                #             input_seg.append(s)
                #             new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #         else:
                #             # 利用tokenizer来校正group_num
                #             lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                #             # print(s)
                #             # print(lm_s)
                #             for ss in lm_s:
                #                 input_seg.append(ss)
                #                 if idx in group_num and g_idx == idx:
                #                     new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #             # for ss in s:
                #             #     input_seg.append(ss)
                #             #     if idx in group_num:
                #             #         new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                # check
                # print(pair['id'])
                graph_seq = ""
                for idx in group_num:
                    if idx < len(input_seq):
                        graph_seq += input_seq[idx]

                lm_graph_seq = ""
                lm_seq = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(' '.join(pair['input_cell'])))
                lm_dict = lm_tokenizer(' '.join(pair['input_cell']))
                lm_seq1 = lm_tokenizer.convert_ids_to_tokens(lm_dict['input_ids'])
                for idx in new_group_num:
                    lm_graph_seq += lm_seq[idx].replace("##", '')
                if len(graph_seq.lower()) != len(lm_graph_seq.lower()) - lm_graph_seq.lower().count('[unk]') * 4:
                    print(pair['id'])
                    print(' '.join(pair['input_cell']))
                    print("group_num:", group_num)
                    print(lm_seq)
                    print(lm_seq1)
                    print("new_group_num:", new_group_num)
                    print(graph_seq.lower())
                    print(lm_graph_seq.lower())
                    print(graph_seq.lower() != lm_graph_seq.lower())
                    print(lm_seq1 != lm_seq)
                    print(len(graph_seq.lower()))
                    print(len(lm_graph_seq.lower()) - lm_graph_seq.count('[unk]') * 4)
                    exit(0)

                group_num_batch.append(new_group_num)

        if use_lm:
            input_batch1 = input_batch
            tokens_dict = lm_tokenizer(input_batch, padding=True, add_special_tokens=False)
            input_batch = []  # tokens_dict["input_ids"]
            #attention_mask_batch = [] # tokens_dict["attention_mask"]
            #token_type_ids_batch = [] # tokens_dict["token_type_ids"]
            attention_mask_batch = tokens_dict["attention_mask"]
            token_type_ids_batch = tokens_dict["token_type_ids"]

            num_pos_batch1 = num_pos_batch
            num_pos_batch = []  # need to be updated, so clear it
            num_size_batch = []
            stage1_span_ids_batch1 = stage1_span_ids_batch
            stage1_span_ids_batch = []
            stage1_span_ids2_batch1 = stage1_span_ids2_batch
            stage1_span_ids2_batch = []
            quantity_indicator_batch = []
            stage1_sentence_ids_batch = []
            cls_loc_batch = []
            cls_loc2_batch = []
            attention_mask_sentence_batch = []
            attention_mask_sentence2_batch = []
            #stage1_span_ids_batch.append(pair['stage1_span'])
            #for input_seq, span_id, mask, type_ids in zip(tokens_dict["input_ids"], stage1_span_ids_batch1, attention_mask_batch, token_type_ids_batch):
            for input_seq, span_id, span_id2 in zip(tokens_dict["input_ids"], stage1_span_ids_batch1, stage1_span_ids2_batch1):
            #for input_seq in tokens_dict["input_ids"]:
                new_seq = []
                new_span_id = []
                new_span_id2 = []
                new_sentence_id = []
                sentence_m = [list(group) for k, group in groupby(input_seq, lambda x: x == 102 or
                                                             x == 0) if not k]
                for i, j in zip(sentence_m, span_id):
                    new_span_id.extend([j] * (len(i)+1))
                
                for i, j in zip(sentence_m, span_id2):
                    new_span_id2.extend([j] * (len(i)+1))
                #For no [sep] token
                #for i, j in zip(sentence_m, span_id):
                #    new_span_id.extend([j] * len(i))
                
                
                #for idx, t_id in enumerate(input_seq):
                #    if t_id == len(lm_tokenizer.vocab):
                #        new_seq.append(1)
                #    elif t_id == 102:
                #        mask.pop(idx)
                #        type_ids.pop(idx)
                #        continue
                #    else:
                #        new_seq.append(t_id)
                #attention_mask_batch.append(mask)
                #token_type_ids_batch.append(type_ids)
                for sentence_id, i in enumerate(sentence_m):
                    new_sentence_id.extend([sentence_id] * (len(i)+1))
                
                for t_id in input_seq:
                    if t_id == len(lm_tokenizer.vocab):
                        new_seq.append(1)
                    else:
                        new_seq.append(t_id)
                input_batch.append(new_seq)
                new_span_id.extend([0] * (input_seq.count(0)))
                new_span_id2.extend([0] * (input_seq.count(0)))
                new_sentence_id.extend([0] * (input_seq.count(0)))
                stage1_span_ids_batch.append(new_span_id)
                stage1_span_ids2_batch.append(new_span_id2)
                stage1_sentence_ids_batch.append(new_sentence_id)
                assert all(len(i) == len(j) for i,j in zip(stage1_span_ids_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(stage1_span_ids2_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(stage1_sentence_ids_batch, input_batch))

            for t_idx, input_seq in enumerate(input_batch):
                num_pos = []
                cls_loc = []
                cls_loc2 = []
                quantity_indicator = []
                for idx, t_id in enumerate(input_seq):
                    # if t_id == lm_tokenizer.vocab['[NUM]']:
                    if t_id == 1:
                        num_pos.append(idx)
                        #quantity_indicator.append(1)
                        if stage1_span_ids_batch[t_idx][idx] == 1:
                            quantity_indicator.append(1)
                        else:
                            quantity_indicator.append(0)
                    elif t_id == 101 and stage1_span_ids_batch[t_idx][idx] == 1 and stage1_span_ids2_batch[t_idx][idx] == 1:
                    #elif t_id == 101:
                        cls_loc.append(idx)
                        cls_loc2.append(idx)
                        quantity_indicator.append(0)
                    elif t_id == 101 and stage1_span_ids_batch[t_idx][idx] == 1:
                    #elif t_id == 101:
                        cls_loc.append(idx)
                        quantity_indicator.append(0)
                    elif t_id == 101 and stage1_span_ids2_batch[t_idx][idx] == 1:
                    #elif t_id == 101:
                        cls_loc2.append(idx)
                        quantity_indicator.append(0)
                    else:
                        quantity_indicator.append(0)
                # if len(num_pos) != len(num_pos_batch1[t_idx]):  # 检查一致性
                #     print(id_batch[t_idx])
                #     print(input_batch1[t_idx])
                #     print(lm_tokenizer.convert_ids_to_tokens(input_batch[t_idx]))
                #     print(len(num_pos))
                #     print(len(num_pos_batch1[t_idx]))
                #     print(group_num_batch[t_idx])
                #     exit(0)

                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                quantity_indicator_batch.append(quantity_indicator)
                cls_loc_padding = pad_seq(cls_loc, len(cls_loc), stage1_len_max, pad_token=-1)
                cls_loc2_padding = pad_seq(cls_loc2, len(cls_loc2), stage1_len_max, pad_token=-1)
                attention_mask_sentence_batch.append([float(i != -1) for i in cls_loc_padding])
                attention_mask_sentence2_batch.append([float(i != -1) for i in cls_loc2_padding])
                cls_loc_batch.append(cls_loc_padding)
                cls_loc2_batch.append(cls_loc2_padding)
                #input_length.append(input_seq.index(lm_tokenizer.vocab['[SEP]'])+1)
                input_length.append(len(input_seq) - input_seq[::-1].index(lm_tokenizer.vocab['[SEP]']))
            attention_mask_batches.append(attention_mask_batch)
            token_type_ids_batches.append(token_type_ids_batch)
            stage1_sentence_ids_batches.append(stage1_sentence_ids_batch)
            stage1_span_ids_batches.append(stage1_span_ids_batch)
            stage1_span_ids2_batches.append(stage1_span_ids2_batch)
            attention_mask_sentence_batches.append(attention_mask_sentence_batch)
            attention_mask_sentence2_batches.append(attention_mask_sentence2_batch)
            #sentence_length_batches.append(sentence_length_batch)
            quantity_indicator_batches.append(quantity_indicator_batch)
            cls_loc_batches.append(cls_loc_batch)
            cls_loc2_batches.append(cls_loc2_batch)
            #prev_eq_batches.append(prev_eq_batch)
            #attention_mask_prev_eq_batches.append(attention_mask_prev_eq_batch)
            
        
        input_batches.append(input_batch)
        input_lengths.append(input_length)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        output2_batches.append(output2_batch)
        num_stack_batches.append(num_stack_batch)
        num_stack2_batches.append(num_stack2_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)

        if use_group_num:
            group_num_batches.append(group_num_batch)
            num_graph_batches.append(get_single_batch_graph(input_batch, input_length, group_num_batch, num_batch, num_pos_batch))

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "output2_batches": output2_batches,
        "output2_lengths": output2_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_stack2_batches": num_stack2_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }

    if use_group_num:
        batches_dict['group_num_batches'] = group_num_batches
        batches_dict['num_graph_batches'] = num_graph_batches

    if use_lm:
        batches_dict['attention_mask_batches'] = attention_mask_batches
        batches_dict['token_type_ids_batches'] = token_type_ids_batches
        batches_dict['stage1_span_ids_batches'] = stage1_span_ids_batches
        batches_dict['stage1_span_ids2_batches'] = stage1_span_ids2_batches
        batches_dict['stage1_sentence_ids_batches'] = stage1_sentence_ids_batches
        batches_dict['quantity_indicator_batches'] = quantity_indicator_batches
        batches_dict['attention_mask_sentence_batches'] = attention_mask_sentence_batches
        batches_dict['attention_mask_sentence2_batches'] = attention_mask_sentence2_batches
        batches_dict['cls_loc_batches'] = cls_loc_batches
        batches_dict['cls_loc2_batches'] = cls_loc2_batches
        batches_dict['var_cnt_batches'] = var_cnt_batches
        batches_dict['equation_id_batches'] = equation_id_batches
        batches_dict['equation_id2_batches'] = equation_id2_batches
        #batches_dict['prev_eq_batches'] = prev_eq_batches
        #batches_dict['prev_eq_len_batches'] = prev_eq_len_batches
        #batches_dict['attention_mask_prev_eq_batches'] = attention_mask_prev_eq_batches
        #batches_dict['stage1_span_lengths'] = stage1_span_lengths

    return batches_dict

def prepare_data_batch_origin_nosep(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0,
                        shuffle=True, use_group_num=False, use_lm=False, lm_tokenizer=None):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle:
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    if use_lm:
        attention_mask_batches = []
        token_type_ids_batches = []
        stage1_span_ids_batches = []
        #stage1_sentence_ids_batches = []
        #attention_mask_sentence_batches = []
        #stage1_span_lengths = []
        #sentence_length_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []

    if use_group_num:
        group_num_batches = []
        num_graph_batches = []

    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        #stage1_span_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len)
        for pair in batch:
            if not use_lm:
                input_length.append(pair["input_cell_len"])
            #else:
            #    stage1_span_length.append(pair["stage1_span_len"])
            output_length.append(pair["output_cell_len"])
                
        # input_lengths.append(input_length)
        output_lengths.append(output_length)
        if not use_lm:
            input_len_max = input_length[0]
        #else:
            #stage1_len_max = max(stage1_span_length)
            #print(stage1_len_max)
            #stage1_span_lengths.append(stage1_span_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        if use_lm:
            stage1_span_ids_batch = []
        if use_group_num:
            group_num_batch = []

        for pair in batch:
            num_batch.append(pair['nums'])
            if use_lm:
                input_batch.append(' '.join(pair['input_cell']))
                stage1_span_ids_batch.append(pair['stage1_span'])
            else:
                input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair["num_stack"])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            if use_group_num and not use_lm:
                group_num_batch.append(pair['group_num'])
            elif use_group_num and use_lm:
                # 要修改
                group_num = pair['group_num']
                input_seq = pair['input_cell']
                new_group_num = []
                pattern = re.compile(r'\[NUM]')

                # update group_num
                acc_count = 0
                temp_input_seq = []
                for idx, s in enumerate(input_seq):
                    if s in ['',  '',  '', '', '', '', '', '']:
                        updated_group_num = []
                        for g_idx in group_num:
                            if g_idx == idx - acc_count:
                                continue
                            elif g_idx > idx - acc_count:
                                updated_group_num.append(g_idx - 1)
                            else:
                                updated_group_num.append(g_idx)
                        acc_count += 1
                        group_num = updated_group_num
                    else:
                        if s != '' and s != '' and s != ' ':
                            temp_input_seq.append(s)
                input_seq = temp_input_seq

                input_seg = []
                seq_mapping = {}
                for idx, s in enumerate(input_seq):
                    pos = re.search(pattern, s)  # 搜索每个词的数字位置
                    if pos and idx in group_num:
                        input_seg.append(s)
                        seq_mapping[idx] = [len(input_seg)-1 + 1]
                    else:
                        seq_mapping[idx] = []
                        # 利用tokenizer来校正group_num
                        lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                        for ss in lm_s:
                            input_seg.append(ss)
                            if idx in group_num:
                                seq_mapping[idx].append(len(input_seg)-1 + 1)
                                # new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                for idx in group_num:
                    if idx < len(input_seq):
                        new_group_num.extend(seq_mapping[idx])

                # for g_idx in group_num:
                #     input_seg = []
                #     for idx, s in enumerate(input_seq):
                #         pos = re.search(pattern, s)  # 搜索每个词的数字位置
                #         if pos and idx in group_num and g_idx == idx:
                #             input_seg.append(s)
                #             new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #         else:
                #             # 利用tokenizer来校正group_num
                #             lm_s = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(s)[1:-1])
                #             # print(s)
                #             # print(lm_s)
                #             for ss in lm_s:
                #                 input_seg.append(ss)
                #                 if idx in group_num and g_idx == idx:
                #                     new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS
                #             # for ss in s:
                #             #     input_seg.append(ss)
                #             #     if idx in group_num:
                #             #         new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

                # check
                # print(pair['id'])
                graph_seq = ""
                for idx in group_num:
                    if idx < len(input_seq):
                        graph_seq += input_seq[idx]

                lm_graph_seq = ""
                lm_seq = lm_tokenizer.convert_ids_to_tokens(lm_tokenizer.encode(' '.join(pair['input_cell'])))
                lm_dict = lm_tokenizer(' '.join(pair['input_cell']))
                lm_seq1 = lm_tokenizer.convert_ids_to_tokens(lm_dict['input_ids'])
                for idx in new_group_num:
                    lm_graph_seq += lm_seq[idx].replace("##", '')
                if len(graph_seq.lower()) != len(lm_graph_seq.lower()) - lm_graph_seq.lower().count('[unk]') * 4:
                    print(pair['id'])
                    print(' '.join(pair['input_cell']))
                    print("group_num:", group_num)
                    print(lm_seq)
                    print(lm_seq1)
                    print("new_group_num:", new_group_num)
                    print(graph_seq.lower())
                    print(lm_graph_seq.lower())
                    print(graph_seq.lower() != lm_graph_seq.lower())
                    print(lm_seq1 != lm_seq)
                    print(len(graph_seq.lower()))
                    print(len(lm_graph_seq.lower()) - lm_graph_seq.count('[unk]') * 4)
                    exit(0)

                group_num_batch.append(new_group_num)

        if use_lm:
            input_batch1 = input_batch
            tokens_dict = lm_tokenizer(input_batch, padding=True, add_special_tokens=False)
            input_batch = []  # tokens_dict["input_ids"]
            attention_mask_batch = [] # tokens_dict["attention_mask"]
            token_type_ids_batch = [] # tokens_dict["token_type_ids"]
            #attention_mask_batch = tokens_dict["attention_mask"]
            #token_type_ids_batch = tokens_dict["token_type_ids"]

            num_pos_batch1 = num_pos_batch
            num_pos_batch = []  # need to be updated, so clear it
            num_size_batch = []
            stage1_span_ids_batch1 = stage1_span_ids_batch
            stage1_span_ids_batch = []

            #stage1_span_ids_batch.append(pair['stage1_span'])
            for input_seq, span_id, mask, type_ids in zip(tokens_dict["input_ids"], stage1_span_ids_batch1, tokens_dict["attention_mask"], tokens_dict["token_type_ids"]):
            #for input_seq, span_id in zip(tokens_dict["input_ids"], stage1_span_ids_batch1):
                new_seq = []
                new_span_id = []
                new_mask = []
                new_type_ids = []
                sentence_m = [list(group) for k, group in groupby(input_seq, lambda x: x == 102 or
                                                             x == 0) if not k]
                #for i, j in zip(sentence_m, span_id):
                #    new_span_id.extend([j] * (len(i)+1))
                
                #For no [sep] token
                for i, j in zip(sentence_m, span_id):
                    new_span_id.extend([j] * len(i))
                
                new_span_id.append(span_id[-1])
                sep_count = input_seq.count(102)
                cnt = 1
                for t_id, t_mask, t_type_ids in zip(input_seq, mask, type_ids):
                    if t_id == len(lm_tokenizer.vocab):
                        new_seq.append(1)
                        new_mask.append(t_mask)
                        new_type_ids.append(t_type_ids)
                    elif t_id == 102 and cnt < sep_count:
                        cnt += 1
                        continue
                    else:
                        new_mask.append(t_mask)
                        new_type_ids.append(t_type_ids)
                        new_seq.append(t_id)
                    
#                 for t_id in input_seq:
#                     if t_id == len(lm_tokenizer.vocab):
#                         new_seq.append(1)
#                     else:
#                         new_seq.append(t_id)
                new_seq.extend([0] * (sep_count-1))
                new_mask.extend([0] * (sep_count-1))
                new_type_ids.extend([0] * (sep_count-1))
    
                input_batch.append(new_seq)
                attention_mask_batch.append(new_mask)
                token_type_ids_batch.append(new_type_ids)
                new_span_id.extend([0] * (new_seq.count(0)))
                stage1_span_ids_batch.append(new_span_id)

                assert all(len(i) == len(j) for i,j in zip(stage1_span_ids_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(attention_mask_batch, input_batch))
                assert all(len(i) == len(j) for i,j in zip(token_type_ids_batch, input_batch))

            for t_idx, input_seq in enumerate(input_batch):
                num_pos = []
                for idx, t_id in enumerate(input_seq):
                    # if t_id == lm_tokenizer.vocab['[NUM]']:
                    if t_id == 1:
                        num_pos.append(idx)
                # if len(num_pos) != len(num_pos_batch1[t_idx]):  # 检查一致性
                #     print(id_batch[t_idx])
                #     print(input_batch1[t_idx])
                #     print(lm_tokenizer.convert_ids_to_tokens(input_batch[t_idx]))
                #     print(len(num_pos))
                #     print(len(num_pos_batch1[t_idx]))
                #     print(group_num_batch[t_idx])
                #     exit(0)

                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                #input_length.append(input_seq.index(lm_tokenizer.vocab['[SEP]'])+1)
                input_length.append(len(input_seq) - input_seq[::-1].index(lm_tokenizer.vocab['[SEP]']))
            attention_mask_batches.append(attention_mask_batch)
            token_type_ids_batches.append(token_type_ids_batch)
            stage1_span_ids_batches.append(stage1_span_ids_batch)
 
            

        input_batches.append(input_batch)
        input_lengths.append(input_length)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)

        if use_group_num:
            group_num_batches.append(group_num_batch)
            num_graph_batches.append(get_single_batch_graph(input_batch, input_length, group_num_batch, num_batch, num_pos_batch))

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }

    if use_group_num:
        batches_dict['group_num_batches'] = group_num_batches
        batches_dict['num_graph_batches'] = num_graph_batches

    if use_lm:
        batches_dict['attention_mask_batches'] = attention_mask_batches
        batches_dict['token_type_ids_batches'] = token_type_ids_batches
        batches_dict['stage1_span_ids_batches'] = stage1_span_ids_batches

    return batches_dict