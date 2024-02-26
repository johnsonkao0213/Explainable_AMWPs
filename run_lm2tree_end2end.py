# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.load_data import *
from src.num_transfer import *
from src.expression_tree import *
from src.log_utils import *
from src.calculation import *
# from src.expressions_transfer import *
from src.data_utils import get_pretrained_embedding_weight, pad_seq
from transformers import BertTokenizer, AdamW
import argparse
from itertools import groupby

torch.cuda.set_device(1)
# USE_CUDA = torch.cuda.is_available()
# batch_size = 16
# grad_acc_steps = 8  # 使用grad_acc_steps步来完成batch_size的训练，每一步：batch_size // grad_acc_steps
# embedding_size = 128
# hidden_size = 768
# n_epochs = 80
# bert_learning_rate = 5e-5
# bert_path = "./pretrained_lm/chinese-bert-wwm"
# learning_rate = 1e-3
# weight_decay = 2e-5
# beam_size = 5
# beam_search = True
# fold_num = 5
# n_layers = 2
# drop_out = 0.5
# random_seed = 1
# var_nums = []
# dataset_name = "mawps"
# ckpt_dir = "Math23K"
# data_path = "../dataset/math23k/Math_23K.json"

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--grad_acc_steps', type=int, default=8)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--n_epochs', type=int, default=80)
parser.add_argument('--stage1_learning_rate', type=float, default=1e-5)
parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
parser.add_argument('--bert_path', type=str, default="./pretrained_lm/chinese-bert-wwm")
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--stage1_weight_decay', type=float, default=5e-6)
parser.add_argument('--weight_decay', type=float, default=2e-5)

parser.add_argument('--enable_beam_search', action='store_true')
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--fold_num', type=int, default=5)

# parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--use_teacher_forcing', type=float, default=1.0)
parser.add_argument('--use_clip', action='store_true')
parser.add_argument('--gclip', type=float, default=0.0)
parser.add_argument('--dataset_name', type=str, default='Math23K')
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
batch_size = args.batch_size  # 16
grad_acc_steps = args.grad_acc_steps  # 8  # 使用grad_acc_steps步来完成batch_size的训练，每一步：batch_size // grad_acc_steps
embedding_size = args.embedding_size  # 128
hidden_size = args.hidden_size  # 768
n_epochs = args.n_epochs  # 80
stage1_learning_rate = args.stage1_learning_rate  # 1e-5
bert_learning_rate = args.bert_learning_rate  # 5e-5
bert_path = args.bert_path  # "./pretrained_lm/chinese-bert-wwm"
learning_rate = args.learning_rate  # 1e-3
weight_decay = args.weight_decay  # 2e-5
stage1_weight_decay = args.stage1_weight_decay  # 5e-6
beam_size = args.beam_size  # 5
beam_search = args.enable_beam_search  # True
fold_num = args.fold_num  # 5
# n_layers = args.n_layers  # 1
drop_out = args.drop_out  # 0.5
random_seed = args.random_seed  # 1
use_clip = args.use_clip
gclip = args.gclip  # 0
var_nums = []
dataset_name = args.dataset_name  # "mawps"
ckpt_dir = "Math23K"
data_path = "../dataset/math23k/Math_23K.json"
num_labels = 5 #category class number
use_sentence_index = True
aggregate_mode = 'mean' #'cls', 'mean', 'sep'
rnn_type = 'transformer'

if dataset_name == "Math23K":
    var_nums = []
    ckpt_dir = "Math23K_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/math23k/Math_23K.json"
elif dataset_name == "Math23K_char":
    var_nums = []
    ckpt_dir = "Math23K_char_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/math23k/Math_23K_char.json"
elif dataset_name == "ALG514":
    var_nums = ['x','y']
    ckpt_dir = "ALG514_b2t_new_retrain6"
    bert_path = "bert-base-uncased"
    data_path = "./dataset/alg514/questions_normalization_v2.json"
    stage1_path = "./benchmark_labels/label_v3_withQ.json"
elif dataset_name == "mawps":
    var_nums = []
    ckpt_dir = "mawps_b2t"
    bert_path = "./pretrained_lm/bert-base-uncased"
    data_path = "./dataset/mawps/mawps_combine.json"
elif dataset_name == "hmwp":
    var_nums = ['x', 'y']
    ckpt_dir = "hmwp_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/hmwp/hmwp.json"
elif dataset_name == "cm17k":
    var_nums = ['x', 'y']
    ckpt_dir = "cm17k_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/cm17k/questions.json"

ckpt_dir = ckpt_dir + '_' + str(n_epochs) + '_' + str(batch_size) + '_' + str(embedding_size) + '_' + str(hidden_size) + \
           'stage1_blr' + str(stage1_learning_rate) + 'stage1_wd' + str(stage1_weight_decay) + \
           '_blr' + str(bert_learning_rate) + '_lr' + str(learning_rate) + '_wd' + str(weight_decay) + '_do' + str(drop_out)
if beam_search:
    ckpt_dir = ckpt_dir + '_' + 'beam_search' + str(beam_size)


save_dir = os.path.join("./models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

for fold_id in range(fold_num):
    if not os.path.exists(os.path.join(save_dir, 'fold-'+str(fold_id))):
        os.mkdir(os.path.join(save_dir, 'fold-'+str(fold_id)))

pairs = None
generate_nums = None
copy_nums = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "Math23K_char":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "ALG514":
    #data = load_alg514_data(data_path)
    data = load_alg514_data(data_path, stage1_path)
    pairs, generate_nums, copy_nums = transfer_alg514_num(data)
elif dataset_name == "mawps":
    data = load_mawps_data(data_path)
    pairs, generate_nums, copy_nums = transfer_mawps_num(data)
elif dataset_name == "hmwp":
    data = load_hmwp_data(data_path)
    pairs, generate_nums, copy_nums = transfer_hmwp_num(data)
elif dataset_name == "cm17k":
    data = load_cm17k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_cm17k_num(data)

temp_pairs = []
for p in pairs:
    ept = ExpressionTree()
    ept.build_tree_from_infix_expression(p["out_seq"])
    p['out_seq'] = ept.get_prefix_expression()
    temp_pairs.append(p)
pairs = temp_pairs

fold_size = int(len(pairs) / fold_num)
fold_pairs = []
for split_fold in range((fold_num - 1)):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * (fold_num - 1)):])

last_acc_fold = []
best_val_acc_fold = []
all_acc_data = []

for fold in range(fold_num):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(fold_num):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    random.seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # cpu
    if USE_CUDA:
        torch.cuda.manual_seed(random_seed) # gpu
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_tokenizer.add_tokens(['[NUM]'])

    _, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                        copy_nums, tree=True, use_lm=True, use_group_num=False)

    embedding_weight = None

    # Initialize models
    stage1_encoder = Stage1_Encoder(num_labels, bert_tokenizer, use_sentence_index, aggregate_mode, rnn_type)
    encoder = PLMEncoderSeq_v2(model_path=bert_path)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                         input_size=len(generate_nums) + len(var_nums), dropout=drop_out)
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                            embedding_size=embedding_size, dropout=drop_out)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=drop_out)

    # the embedding layer is  only for generated number embeddings, operators, and paddings
    stage1_encoder_optimizer = AdamW(stage1_encoder.parameters(), lr=stage1_learning_rate, weight_decay=stage1_weight_decay)
    encoder_optimizer = AdamW(encoder.parameters(), lr=bert_learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    stage1_encoder_scheduler = torch.optim.lr_scheduler.StepLR(stage1_encoder_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        stage1_encoder.cuda()
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    var_num_ids = []
    for var in var_nums:
        if var in output_lang.word2index.keys():
            var_num_ids.append(output_lang.word2index[var])

    best_val_acc = 0
    best_equ_acc = 0
    current_save_dir = os.path.join(save_dir, 'fold-'+str(fold))
    current_best_val_acc = (0,0,0)
    for epoch in range(n_epochs):
        loss_total = 0
        stage1_loss_total = 0
        sent_acc_total, pro_acc_total, problem_len_total = 0, 0, 0
        sentence_len_total = 0
        random.seed(epoch+random_seed)  # for reproduction
        batch_dict = prepare_data_batch(train_pairs, batch_size, lm_tokenizer=bert_tokenizer,
                                        use_group_num=False, use_lm=True)

        id_batches = batch_dict['id_batches']
        input_batches = batch_dict['input_batches']
        input_lengths = batch_dict['input_lengths']
        attention_mask_batches = batch_dict['attention_mask_batches']
        token_type_ids_batches = batch_dict['token_type_ids_batches']
        output_batches = batch_dict['output_batches']
        output_lengths = batch_dict['output_lengths']
        nums_batches = batch_dict['nums_batches']
        num_stack_batches = batch_dict['num_stack_batches']
        num_pos_batches = batch_dict['num_pos_batches']
        num_size_batches = batch_dict['num_size_batches']
        ans_batches = batch_dict['ans_batches']
        stage1_span_ids_batches = batch_dict['stage1_span_ids_batches']
        stage1_sentence_ids_batches = batch_dict['stage1_sentence_ids_batches']
        attention_mask_sentence_batches = batch_dict['attention_mask_sentence_batches']
        stage1_span_lengths = batch_dict['stage1_span_lengths']
        sentence_length_batches = batch_dict['sentence_length_batches']

        logs_content = "fold: {}".format(fold+1)
        add_log(log_file, logs_content)
        logs_content = "epoch: {}".format(epoch + 1)
        add_log(log_file,logs_content)
        start = time.time()
        for idx in range(len(input_lengths)):
            step_size = len(input_batches[idx]) // grad_acc_steps
            for step in range(grad_acc_steps):
                start_idx = step * step_size
                end_idx = (step + 1) * step_size
                if step_size == 0:
                    end_idx = len(input_batches[idx])

                if step == grad_acc_steps - 1:
                    grad_acc = False
                else:
                    grad_acc = True

                if step == 0:
                    zero_grad = True
                else:
                    zero_grad = False

                loss, stage1_loss, sent_acc, pro_acc, problem_len = train_lm2tree_v2(input_batches[idx][start_idx:end_idx],
                                     input_lengths[idx][start_idx:end_idx],
                                     attention_mask_batches[idx][start_idx:end_idx],
                                     token_type_ids_batches[idx][start_idx:end_idx],
                                     output_batches[idx][start_idx:end_idx],
                                     output_lengths[idx][start_idx:end_idx],
                                     num_stack_batches[idx][start_idx:end_idx],
                                     num_size_batches[idx][start_idx:end_idx],
                                     stage1_span_ids_batches[idx][start_idx:end_idx],
                                     stage1_span_lengths[idx][start_idx:end_idx],
                                     stage1_sentence_ids_batches[idx][start_idx:end_idx],
                                     attention_mask_sentence_batches[idx][start_idx:end_idx],
                                     sentence_length_batches[idx][start_idx:end_idx],
                                     generate_num_ids, stage1_encoder, encoder, predict, generate, merge,
                                     stage1_encoder_optimizer, encoder_optimizer, predict_optimizer, 
                                     generate_optimizer, merge_optimizer, output_lang,
                                     num_pos_batches[idx][start_idx:end_idx], id_batches[idx][start_idx:end_idx], 
                                     use_clip=use_clip, clip=gclip,
                                     grad_acc=grad_acc, zero_grad=zero_grad, grad_acc_steps=grad_acc_steps,
                                     var_nums=var_num_ids, labels_num=num_labels)
                loss_total += loss
                stage1_loss_total += stage1_loss
                sent_acc_total += sent_acc
                pro_acc_total +=  pro_acc
                problem_len_total += problem_len
                sentence_len_total += step_size
        
        stage1_encoder_scheduler.step()
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        
        logs_content = "stage1 loss: {}".format(stage1_loss_total / len(input_lengths))
        add_log(log_file,logs_content)
        logs_content = "Training stage1 Sentence Accuracy: {}".format(sent_acc_total / problem_len_total)
        add_log(log_file, logs_content)
        logs_content = "Training stage1 Problem Accuracy: {}".format(pro_acc_total / sentence_len_total)
        add_log(log_file, logs_content)
        logs_content = "loss: {}".format(loss_total / len(input_lengths))
        add_log(log_file,logs_content)
        logs_content = "training time: {}".format(time_since(time.time() - start))
        add_log(log_file,logs_content)
        logs_content = "--------------------------------"
        add_log(log_file,logs_content)
        if epoch % 1 == 0 or epoch > n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            answer_ac = 0
            eval_total = 0
            test_sent_acc_total, test_pro_acc_total, test_problem_len_total = 0, 0, 0
            start = time.time()
            for test_batch in test_pairs:
                tokens_dict = bert_tokenizer(' '.join(test_batch['input_cell']), add_special_tokens=False)
                raw_input_ids = tokens_dict["input_ids"]
                tokens_dict["input_ids"] = []
                stage1_sentence_ids = []
                sentence_length = []
                sentence_m = [list(group) for k, group in groupby(raw_input_ids, lambda x: x == 102 or
                                                             x == 0) if not k]
                for sentence_id, i in enumerate(sentence_m):
                    sentence_length.append(len(i)+1)
                    stage1_sentence_ids.extend([sentence_id] * (len(i)+1))
                    
                stage1_sentence_ids.extend([0] * (raw_input_ids.count(0)))
                
                for t_id in raw_input_ids:
                    if t_id == len(bert_tokenizer.vocab):
                        tokens_dict["input_ids"].append(1)
                    else:
                        tokens_dict["input_ids"].append(t_id)

                num_pos = []
                for idx, t_id in enumerate(tokens_dict["input_ids"]):
                    if t_id == 1:
                        num_pos.append(idx)

                attention_mask_sentence = [float(i != -100) for i in test_batch['stage1_span']]
                print(test_batch['id'])
                test_res, test_sent_acc, test_pro_acc, test_problem_len = evaluate_lm2tree_v2(tokens_dict["input_ids"], 
                                               len(tokens_dict["input_ids"]), 
                                               tokens_dict["attention_mask"], tokens_dict["token_type_ids"], 
                                               test_batch['stage1_span'], test_batch['stage1_span_len'],
                                               stage1_sentence_ids, attention_mask_sentence, sentence_length,
                                               generate_num_ids, stage1_encoder, encoder, predict, generate,
                                               merge, output_lang, num_pos, beam_size=beam_size, beam_search=beam_search,
                                               var_nums=var_num_ids, labels_num=num_labels)
                
                test_sent_acc_total += test_sent_acc
                test_pro_acc_total += test_pro_acc
                test_problem_len_total += test_problem_len
                
                import traceback
                try:
                    val_ac, equ_ac, ans_ac, \
                    test_res_result, test_tar_result = compute_equations_result(test_res, test_batch['output_cell'],
                                                                                output_lang, test_batch['nums'],
                                                                                test_batch['num_stack'],
                                                                                ans_list=test_batch['ans'],
                                                                                tree=True, prefix=True)
                    # print(test_res_result, test_tar_result)
                except Exception as e:
                    # traceback.print_exc()
                    # print(e)
                    val_ac, equ_ac, ans_ac = False, False, False
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                if ans_ac:
                    answer_ac += 1
                eval_total += 1

            logs_content = "Testing Stage1 Sentence Accuracy: {}".format(test_sent_acc_total / test_problem_len_total)
            add_log(log_file, logs_content)
            logs_content = "Testing Stage1 Problem Accuracy: {}".format(test_pro_acc_total / len(test_pairs))
            add_log(log_file, logs_content)
            logs_content = "{}, {}, {}".format(equation_ac, value_ac, eval_total)
            add_log(log_file, logs_content)
            logs_content = "test_answer_acc: {} {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total)
            add_log(log_file, logs_content)
            logs_content = "testing time: {}".format(time_since(time.time() - start))
            add_log(log_file, logs_content)
            logs_content = "------------------------------------------------------"
            add_log(log_file, logs_content)
            all_acc_data.append((fold, epoch,equation_ac, value_ac, eval_total))

            torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder"))
            torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict"))
            torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate"))
            torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge"))
            if best_val_acc < value_ac:
                best_val_acc = value_ac
                current_best_val_acc = (equation_ac, value_ac, eval_total)
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_val_acc"))
                torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_val_acc"))
                torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_val_acc"))
                torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_val_acc"))
            if best_equ_acc < equation_ac:
                best_equ_acc = equation_ac
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_equ_acc"))
                torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_equ_acc"))
                torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_equ_acc"))
                torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_equ_acc"))
            if epoch == n_epochs - 1:
                last_acc_fold.append((equation_ac, value_ac, eval_total))
                best_val_acc_fold.append(current_best_val_acc)

a, b, c = 0, 0, 0
for bl in range(len(last_acc_fold)):
    a += last_acc_fold[bl][0]
    b += last_acc_fold[bl][1]
    c += last_acc_fold[bl][2]
    logs_content = "{}".format(last_acc_fold[bl])
    add_log(log_file, logs_content)
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

a, b, c = 0, 0, 0
for bl in range(len(best_val_acc_fold)):
    a += best_val_acc_fold[bl][0]
    b += best_val_acc_fold[bl][1]
    c += best_val_acc_fold[bl][2]
    logs_content = "{}".format(best_val_acc_fold[bl])
    add_log(log_file, logs_content)
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

logs_content = "{}".format(all_acc_data)
add_log(log_file, logs_content)
