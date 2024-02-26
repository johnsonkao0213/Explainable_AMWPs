from src.masked_cross_entropy import *
from src.prepare_data import *
from src.expression_tree import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()
PAD_token = 0


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english, var_nums=[]):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums + 2).fill_(-float("1e12"))
    if english:
        # if decoder_input[0] == word2index["[SOS]"]:
        #     for i in range(batch_size):
        #         res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] +  \
        #               [word2index["("]] + generate_nums + var_nums
        #         for j in res:
        #             rule_mask[i, j] = 0
        #     return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["[SOS]"]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                      [word2index["("]] + generate_nums + var_nums
            elif ("[SEP]" in word2index.keys() and decoder_input[i] == word2index["[SEP]"]) or \
                    ("=" in word2index.keys() and decoder_input[i] == word2index['=']):
                res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                      generate_nums + var_nums + [word2index["("]]
            elif decoder_input[i] >= nums_start: # N1 ... Nx
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] == word2index["[EOS]"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                       [word2index["("]] + generate_nums + var_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + [word2index["("]] + \
                       generate_nums + var_nums
            elif decoder_input[i] in var_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif '^' in word2index.keys() and decoder_input[i] == word2index['^']:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums

            for j in res:
                rule_mask[i, j] = 0
    else:
        # if decoder_input[0] == word2index["[SOS]"]:
        #     for i in range(batch_size):
        #         res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
        #               [word2index["["], word2index["("]] + generate_nums
        #         for j in res:
        #             rule_mask[i, j] = 0
        #     return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["[SOS]"]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                      [word2index["("]] + generate_nums + var_nums
            elif ("[SEP]" in word2index.keys() and decoder_input[i] == word2index["[SEP]"]) or \
                    ("=" in word2index.keys() and decoder_input[i] == word2index['=']):
                res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                      generate_nums + var_nums + [word2index["("]]
            elif decoder_input[i] >= nums_start: # N1 ... Nx
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] == word2index["[EOS]"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                       [word2index["("]] + generate_nums + var_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["[EOS]"]]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + \
                       [word2index["["], word2index["("]] + generate_nums + var_nums
            elif decoder_input[i] in var_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["[EOS]"]
                        ]
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif '^' in word2index.keys() and decoder_input[i] == word2index['^']:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums

            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english, var_nums=[]):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        # if decoder_input[0] == word2index["[SOS]"]:
        #     for i in range(batch_size):
        #         res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
        #               [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
        #         for j in res:
        #             rule_mask[i, j] = 0
        #     return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["[SOS]"]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] >= nums_start:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] == word2index["[EOS]"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in var_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif '^' in word2index.keys() and decoder_input[i] == word2index['^']:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums

            for j in res:
                rule_mask[i, j] = 0
    else:
        # if decoder_input[0] == word2index["[SOS]"]:
        #     for i in range(batch_size):
        #         res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
        #               [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
        #         for j in res:
        #             rule_mask[i, j] = 0
        #     return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["[SOS]"]:
                res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
                if len(var_nums) > 0 and "=" in word2index.keys():
                    res += [word2index['=']]
                if len(var_nums) > 0 and "[SEP]" in word2index.keys():
                    res += [word2index["[SEP]"]]
            elif decoder_input[i] >= nums_start:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] == word2index["[EOS]"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in var_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif '^' in word2index.keys() and decoder_input[i] == word2index['^']:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums

            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english, var_nums=[]):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        # if decoder_input[0] == word2index["[SOS]"]:
        #     for i in range(batch_size):
        #         res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums
        #         for j in res:
        #             rule_mask[i, j] = 0
        #     return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["[SOS]"]:
                res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + var_nums
            elif decoder_input[i] >= nums_start:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] == word2index["[EOS]"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in var_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif '^' in word2index.keys() and decoder_input[i] == word2index['^']:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                # if '^' in word2index.keys():
                #     res += [word2index['^']]
            for j in res:
                rule_mask[i, j] = 0
    else:
        # if decoder_input[0] == word2index["[SOS]"]:
        #     for i in range(batch_size):
        #         res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums
        #         for j in res:
        #             rule_mask[i, j] = 0
        #     return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["[SOS]"]:
                res = [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + var_nums
            elif decoder_input[i] >= nums_start:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] == word2index["[EOS]"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif decoder_input[i] in var_nums:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]] + var_nums
                if '^' in word2index.keys():
                    res += [word2index['^']]
            elif '^' in word2index.keys() and decoder_input[i] == word2index['^']:
                res += [_ for _ in range(nums_start, nums_start + len(nums_batch[i]))] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["[EOS]"]
                        ] + var_nums
                # if '^' in word2index.keys():
                #     res += [word2index['^']]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            # if len(num_stack) > 1:
            #     target[i] = num_stack[0] + num_start
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()

    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    # 替换了unk符的方程等式
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index) # B x num_size x H
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() # S x B x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0) # 屏蔽其他无关数字


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal
        

        
#gumbel + quantity indicator + end2end
def train_lm2tree_v6(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, 
                  nums_stack_batch, num_size_batch, stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                  attention_mask_sentence_batch, sentence_length_batch, quantity_indicator_batch, var_cnt_batch, 
                  generate_nums, encoder, predict, generate, merge, 
                  encoder_optimizer, predict_optimizer, generate_optimizer,
                  merge_optimizer, output_lang, num_pos, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5, predict_v1=None, 
                  generate_v1=None, merge_v1=None, predict_v1_optimizer=None, generate_v1_optimizer=None, 
                  merge_v1_optimizer=None):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums)  # 最大的位置列表数目+常识数字数目+未知数列表
    for i in num_size_batch:
        d = i + len(generate_nums) + len(var_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    quantity_ids_var = torch.LongTensor(quantity_indicator_batch)
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    quantity_ids_var = quantity_ids_var[:, :max_len]
    
    #stage1 label
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    #attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    #attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch)
    #sep_loc_var = torch.LongTensor(sep_loc_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    #attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    #sep_loc_var = sep_loc_var[:, :max_stage1_len]
    #stage1_span_length_var = torch.LongTensor(stage1_span_length)
    
    masks_1 = (stage1_span_ids_var == 1) | (stage1_span_ids_var == 3) | (stage1_span_ids_var == 4)
    masks_2 = (stage1_span_ids_var == 2) | (stage1_span_ids_var == 3) | (stage1_span_ids_var == 4)
    stage1_span_ids_var_1 = masks_1.long()
    stage1_span_ids_var_2 = masks_2.long()
    
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计
    
    target_v1 = torch.LongTensor(target_batch).transpose(0, 1)
    target_v1 = target_v1[:max_target_len, :]  # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    padding_hidden_v1 = torch.FloatTensor([0.0 for _ in range(predict_v1.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    predict_v1.train()
    generate_v1.train()
    merge_v1.train()


    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        quantity_ids_var = quantity_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        padding_hidden_v1 = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
    
    for var_cnt in var_count_tensor:
        if var_cnt==1:
            encoder_outputs, problem_output = encoder(input_var, attention_mask_var, stage1_span_ids_var) 
            # Run words through encoder
            encoder_outputs = encoder_outputs.transpose(0,1)
            # Prepare input and output variables
            node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

            max_target_length = max(target_length)

            all_node_outputs = []
            # all_leafs = []

            copy_num_len = [len(_) for _ in num_pos]
            num_size = max(copy_num_len)
            all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                      encoder.config.hidden_size)

            num_start = output_lang.num_start - len(var_nums)
            embeddings_stacks = [[] for _ in range(batch_size)]
            left_childs = [None for _ in range(batch_size)]
            for t in range(max_target_length):
                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

                # all_leafs.append(p_leaf)
                outputs = torch.cat((op, num_score), 1)
                all_node_outputs.append(outputs)

                target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, 
                                                               copy.deepcopy(nums_stack_batch), num_start, unk)
                target[t] = target_t
                if USE_CUDA:
                    generate_input = generate_input.cuda()
                left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                       node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue

                    if i < num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                    else:
                        current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, terminal=True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)

            # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
            all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

            target = target.transpose(0, 1).contiguous()
            if USE_CUDA:
                # all_leafs = all_leafs.cuda()
                all_node_outputs = all_node_outputs.cuda()
                target = target.cuda()
        else:
            encoder_outputs, problem_output = encoder(input_var, attention_mask_var, stage1_span_ids_var_1) 
            # Run words through encoder
            encoder_outputs = encoder_outputs.transpose(0,1)
            # Prepare input and output variables
            node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

            max_target_length = max(target_length)

            all_node_outputs = []
            # all_leafs = []

            copy_num_len = [len(_) for _ in num_pos]
            num_size = max(copy_num_len)
            all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                      encoder.config.hidden_size)

            num_start = output_lang.num_start - len(var_nums)
            embeddings_stacks = [[] for _ in range(batch_size)]
            left_childs = [None for _ in range(batch_size)]
            for t in range(max_target_length):
                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

                # all_leafs.append(p_leaf)
                outputs = torch.cat((op, num_score), 1)
                all_node_outputs.append(outputs)

                target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, 
                                                               copy.deepcopy(nums_stack_batch), num_start, unk)
                target[t] = target_t
                if USE_CUDA:
                    generate_input = generate_input.cuda()
                left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                       node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue

                    if i < num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                    else:
                        current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, terminal=True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)

            # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
            all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
            
            #v1
            encoder_outputs_v1, problem_output_v1 = encoder(input_var, attention_mask_var, stage1_span_ids_var_2) 
            # Run words through encoder
            encoder_outputs_v1 = encoder_outputs_v1.transpose(0,1)
            # Prepare input and output variables
            node_stacks_v1 = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

            max_target_length = max(target_length)

            all_node_outputs_v1 = []
            # all_leafs = []

            copy_num_len = [len(_) for _ in num_pos]
            num_size = max(copy_num_len)
            all_nums_encoder_outputs_v1 = get_all_number_encoder_outputs(encoder_outputs_v1, num_pos, batch_size, num_size,
                                                                      encoder.config.hidden_size)

            num_start = output_lang.num_start - len(var_nums)
            embeddings_stacks_v1 = [[] for _ in range(batch_size)]
            left_childs_v1 = [None for _ in range(batch_size)]
            
            for t in range(max_target_length):
                num_score_v1, op_v1, current_embeddings_v1, current_context_v1, current_nums_embeddings_v1 = predict_v1(
                    node_stacks_v1, left_childs_v1, encoder_outputs_v1, all_nums_encoder_outputs_v1, padding_hidden_v1, seq_mask, num_mask)

                # all_leafs.append(p_leaf)
                outputs_v1 = torch.cat((op_v1, num_score_v1), 1)
                all_node_outputs_v1.append(outputs_v1)

                target_t_v1, generate_input_v1 = generate_tree_input(target_v1[t].tolist(), outputs_v1, copy.deepcopy(nums_stack_batch), num_start, unk)
                target_v1[t] = target_t_v1
                if USE_CUDA:
                    generate_input_v1 = generate_input_v1.cuda()
                left_child_v1, right_child_v1, node_label_v1 = generate_v1(current_embeddings_v1, generate_input_v1, current_context_v1)
                left_childs_v1 = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child_v1.split(1), right_child_v1.split(1),
                                               node_stacks_v1, target_v1[t].tolist(), embeddings_stacks_v1):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs_v1.append(None)
                        continue

                    if i < num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label_v1[idx].unsqueeze(0), terminal=False))
                    else:
                        current_num = current_nums_embeddings_v1[idx, i - num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = merge_v1(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, terminal=True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs_v1.append(o[-1].embedding)
                    else:
                        left_childs_v1.append(None)

            # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
            all_node_outputs_v1 = torch.stack(all_node_outputs_v1, dim=1)  # B x S x N

            target = target.transpose(0, 1).contiguous()
            target_v1 = target_v1.transpose(0, 1).contiguous()
            if USE_CUDA:
                # all_leafs = all_leafs.cuda()
                all_node_outputs = all_node_outputs.cuda()
                target = target.cuda()
                all_node_outputs_v1 = all_node_outputs_v1.cuda()
                target_v1= target_v1.cuda()
                
                
        
        
    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss_eq = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    tr_loss = tr_loss / grad_acc_steps
    #loss = loss_eq + (0.01 * tr_loss)
    loss = loss_eq + tr_loss
    #tr_loss.backward(retain_graph=True)
    loss.backward()
    #loss_eq.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        #torch.nn.utils.clip_grad_norm_(stage1_encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        #stage1_encoder_optimizer.step()
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
    
    return loss_eq.item(), tr_loss.item(), tr_cor, tr_acc, tr_label_l  # , loss_0.item(), loss_1.item()
    
    
def evaluate_lm2tree_v5(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, quantity_indicator_batch, sep_loc_batch, generate_nums, 
                       encoder, predict, generate, merge, output_lang, num_pos,
                       beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    quantity_ids_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    #print(stage1_span_ids_batch)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    #attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch).unsqueeze(0)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)
    sep_loc_var = torch.LongTensor(sep_loc_batch).unsqueeze(0)
    #sentence_length_var = torch.LongTensor([sentence_length_batch]).unsqueeze(0)
    #stage1_span_length_var = torch.LongTensor([len(stage1_span_ids_batch)]).unsqueeze(0)
    #print(stage1_span_length_var)
    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        quantity_ids_var = quantity_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        sep_loc_var = sep_loc_var.cuda()
        #stage1_span_length_var = stage1_span_length_var.cuda()
        #sentence_length_var = sentence_length_var.cuda()
    
    #Stage1 span classification
    #sep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)
    #nosep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var, [sentence_length_batch])
    
    tr_logits, tr_loss, encoder_outputs, problem_output = encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_ids_var, stage1_span_ids_var, [sentence_length_batch], [stage1_span_length], sep_loc_var)
    

    # compute training accuracy
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
    tr_acc = 0
    if labels.tolist() == predictions.tolist():
        tr_acc+=1

    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    
    
    # Run words through encoder
    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    #encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
    #                                          token_type_ids=token_type_ids_var, sentence_ids=stage1_sentence_ids_var,
    #                                          stage1_ids=extended_predictions_var)
    encoder_outputs = encoder_outputs.transpose(0,1)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.stage2.config.hidden_size)
    num_start = output_lang.num_start - len(var_nums)
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    if beam_search:
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), terminal=False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, terminal=True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out, tr_cor, tr_acc, tr_label_l
    else:
        all_node_outputs = []
        for t in range(max_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_scores = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            out_tokens = torch.argmax(out_scores, dim=1) # B
            all_node_outputs.append(out_tokens)
            left_childs = []
            flag = False
            for idx, node_stack, out_token, embeddings_stack in zip(range(batch_size), node_stacks, out_tokens, embeddings_stacks):
                # node = node_stack.pop()
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    flag = True
                    break
                    # continue
                # var_num当时数字处理，SEP/;当操作符处理
                if out_token < num_start: # 非数字
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                    node_stack.append(TreeNode(right_child))
                    node_stack.append(TreeNode(left_child, left_flag=True))
                    embeddings_stack.append(TreeEmbedding(node_label.unsqueeze(0), terminal=False))
                else: # 数字
                    current_num = current_nums_embeddings[idx, out_token - num_start].unsqueeze(0)
                    while len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
                        sub_stree = embeddings_stack.pop()
                        op = embeddings_stack.pop()
                        current_num = merge(op.embedding.squeeze(0), sub_stree.embedding, current_num)
                    embeddings_stack.append(TreeEmbedding(current_num, terminal=True))

                if len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
                    left_childs.append(embeddings_stack[-1].embedding)
                else:
                    left_childs.append(None)

            if flag:
                break

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        all_node_outputs = all_node_outputs.cpu().numpy()
        return all_node_outputs[0], tr_cor, tr_acc, tr_label_l