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
#USE_CUDA = False
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
    #def __init__(self, score, node_stack, embedding_stack, left_childs, out, target_node):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        #self.target_node = copy_list(target_node)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_seq2seq(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
                  generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, use_clip=False,
                  clip=0, use_teacher_forcing=1, scheduled_sampling=False, beam_size=1, beam_search=True, var_nums=[],
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    # num_start = output_lang.n_words - copy_nums - 2
    num_start = output_lang.num_start - len(var_nums)
    unk = output_lang.word2index["[UNK]"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :] # 因为梯度累计

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]] * batch_size)

    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if scheduled_sampling:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            if random.random() < use_teacher_forcing:
                decoder_input = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                target[t] = decoder_input
            else:
                target[t] = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
                decoder_input = torch.argmax(decoder_output, dim=1)

    elif random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
            target[t] = decoder_input
    elif beam_search:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                #                                num_start, copy_nums, generate_nums, english, var_nums=var_nums)
                decoder_input = decoder_input.unsqueeze(0)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) #+ rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output

            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos.long() * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
    else:
        # greedy search
        for t in range(max_target_length):
            # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
            #                                num_start, copy_nums, generate_nums, english, var_nums=var_nums)
            decoder_input = decoder_input.unsqueeze(0)

            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)

            all_decoder_outputs[t] = decoder_output
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
            decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            # all_decoder_outputs[t] = decoder_output
            decoder_input = torch.argmax(decoder_output, dim=1)


    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq * class_szie
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    ) / grad_acc_steps

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    # if clip:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        decoder_optimizer.step()

    return return_loss


def evaluate_seq2seq(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                     beam_size=1, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]])  # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    if beam_search:
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

        # Run through decoder
        for di in range(max_length):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == output_lang.word2index["[EOS]"]:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0].all_output
            beam_scores = torch.zeros(decoder.output_size * beam_len)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == output_lang.word2index["[EOS]"]:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
                #                                1, num_start, copy_nums, generate_nums, english, var_nums=var_nums)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_input = decoder_input.unsqueeze(0)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
                score = f.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(beam_size)

            for k in range(beam_size):
                word_n = int(topi[k])
                word_input = word_n % decoder.output_size
                temp_input = torch.LongTensor([word_input])
                indices = int(word_n / decoder.output_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices]+[word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:beam_size]
        all_outputs = beam_list[0].all_output
    else:
        all_outputs = []
        for di in range(max_length):
            # if batch_size == 1:
            #     rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            # else:
            #     rule_mask = generate_rule_mask(decoder_input, num_list, output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_input = decoder_input.unsqueeze(0)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            decoder_output = f.log_softmax(decoder_output, dim=1)
            # all_outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=1)
            all_outputs.append(decoder_input)
        all_outputs = torch.stack(all_outputs, dim=1)  # B x S x N
        all_outputs = all_outputs.cpu().detach().numpy()[0]

    return all_outputs


def train_lm2seq(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length,
                 num_batch, nums_stack_batch, stage1_span_ids_batch, copy_nums, generate_nums,
                 encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang,
                 use_clip=False, clip=0,
                 use_teacher_forcing=1, scheduled_sampling=False, beam_size=1, beam_search=True, var_nums=[],
                 grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)



    # num_start = output_lang.n_words - copy_nums - 2
    num_start = output_lang.num_start - len(var_nums)
    unk = output_lang.word2index["[UNK]"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_span_ids_var = stage1_span_ids_var[:, :max_len]

    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计
    #print("target ground truth")
    #print(target)
    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, stage1_ids=stage1_span_ids_var)
    encoder_outputs = encoder_outputs.transpose(0,1)
    encoder_hidden = encoder_hidden.unsqueeze(0)

    # print(encoder_hidden.size())
    # print(encoder_outputs.size())

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]] * batch_size)

    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    #print(decoder_hidden.size())

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if scheduled_sampling:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            if random.random() < use_teacher_forcing:
                decoder_input = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                target[t] = decoder_input
            else:
                target[t] = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
                decoder_input = torch.argmax(decoder_output, dim=1)

    elif random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
            target[t] = decoder_input
            #print("decoder_inupt")
            #print(decoder_input)
    elif beam_search:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                #                                num_start, copy_nums, generate_nums, english, var_nums=var_nums)
                decoder_input = decoder_input.unsqueeze(0)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) #+ rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output

            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos.int() * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
    else:
        # greedy search
        for t in range(max_target_length):
            # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
            #                                num_start, copy_nums, generate_nums, english, var_nums=var_nums)
            decoder_input = decoder_input.unsqueeze(0)

            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)

            all_decoder_outputs[t] = decoder_output
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
            decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            # all_decoder_outputs[t] = decoder_output
            decoder_input = torch.argmax(decoder_output, dim=1)
    #print("Target")
    #print(target.transpose(0, 1).contiguous())
    # Loss calculation and backpropagation
    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq * class_szie
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    ) / grad_acc_steps

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    # if clip:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        decoder_optimizer.step()

    return return_loss


def evaluate_lm2seq(input_seq, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                     beam_size=1, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, stage1_ids=stage1_span_ids_var)
    encoder_outputs = encoder_outputs.transpose(0,1)
    encoder_hidden = encoder_hidden.unsqueeze(0)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]])  # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    if beam_search:
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

        # Run through decoder
        for di in range(max_length):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == output_lang.word2index["[EOS]"]:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0].all_output
            beam_scores = torch.zeros(decoder.output_size * beam_len)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == output_lang.word2index["[EOS]"]:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
                #                                1, num_start, copy_nums, generate_nums, english, var_nums=var_nums)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_input = decoder_input.unsqueeze(0)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
                score = f.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(beam_size)

            for k in range(beam_size):
                word_n = int(topi[k])
                word_input = word_n % decoder.output_size
                temp_input = torch.LongTensor([word_input])
                indices = int(word_n / decoder.output_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices]+[word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:beam_size]
        all_outputs = beam_list[0].all_output
    else:
        all_outputs = []
        for di in range(max_length):
            # if batch_size == 1:
            #     rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            # else:
            #     rule_mask = generate_rule_mask(decoder_input, num_list, output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_input = decoder_input.unsqueeze(0)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            decoder_output = f.log_softmax(decoder_output, dim=1)
            # all_outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=1)
            all_outputs.append(decoder_input)
        all_outputs = torch.stack(all_outputs, dim=1)  # B x S x N
        all_outputs = all_outputs.cpu().detach().numpy()[0]

    return all_outputs



def train_lm2seq_v2(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length,
                 num_batch, nums_stack_batch, stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                 attention_mask_sentence_batch, sentence_length_batch, id_batch, copy_nums, generate_nums,
                 stage1_encoder, encoder, decoder, stage1_encoder_optimizer, encoder_optimizer, decoder_optimizer, 
                 output_lang, use_clip=False, clip=0,
                 use_teacher_forcing=1, scheduled_sampling=False, beam_size=1, beam_search=True, var_nums=[],
                 grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5, pipeline=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    #print(id_batch)

    # num_start = output_lang.n_words - copy_nums - 2
    num_start = output_lang.num_start - len(var_nums)
    unk = output_lang.word2index["[UNK]"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    
    #stage1 label
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]

    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计
    
    batch_size = len(input_length)
    
    #if not pipeline:
    stage1_encoder.train()
    #else:
    #    for param in stage1_encoder.parameters():
    #        param.requires_grad = False
    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        #if not pipeline:
        stage1_encoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    #Stage1 span classification
    stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)

    # compute training accuracy
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    tr_logits = stage1_outputs.logits
    tr_loss = stage1_outputs.loss
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    
    #continuous value
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
    #print(flattened_predictions.shape)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    #extend stage1 prediction to word unit length
    #extended_predictions = extend_prediction_span(input_var, predictions, stage1_span_length)
    extended_predictions = []
    l = 0
    tr_acc = 0
    for i,j in enumerate(stage1_span_length):
        label_e = labels.tolist()[l:(l+j)]
        prediction_e = predictions.tolist()[l:(l+j)]
        l+=j
        assert j == len(sentence_length_batch[i])
        if label_e == prediction_e:
            tr_acc+=1
        extended_prediction = []
        for k,v in zip(prediction_e, sentence_length_batch[i]):
            extended_prediction.extend([k] * v)
        extended_prediction.extend([0] * (max_len-len(extended_prediction)))
        extended_predictions.append(extended_prediction)
    
    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    
    extended_predictions_var = torch.LongTensor(extended_predictions)
    extended_predictions_var = extended_predictions_var[:, :max_len]
    assert extended_predictions_var.shape == input_var.shape
    if USE_CUDA:
        extended_predictions_var = extended_predictions_var.cuda()
    # Run words through encoder
    #print(attention_mask_var)
    #print(extended_predictions)
    encoder_outputs, encoder_hidden = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, stage1_span_ids=extended_predictions_var)
    encoder_outputs = encoder_outputs.transpose(0,1)
    encoder_hidden = encoder_hidden.unsqueeze(0)

    # print(encoder_hidden.size())
    # print(encoder_outputs.size())

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]] * batch_size)

    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    #print(decoder_hidden.size())

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if scheduled_sampling:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            if random.random() < use_teacher_forcing:
                decoder_input = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                target[t] = decoder_input
            else:
                target[t] = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
                decoder_input = torch.argmax(decoder_output, dim=1)

    elif random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
            target[t] = decoder_input
    elif beam_search:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                #                                num_start, copy_nums, generate_nums, english, var_nums=var_nums)
                decoder_input = decoder_input.unsqueeze(0)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) #+ rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output

            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos.int() * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
    else:
        # greedy search
        for t in range(max_target_length):
            # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
            #                                num_start, copy_nums, generate_nums, english, var_nums=var_nums)
            decoder_input = decoder_input.unsqueeze(0)

            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)

            all_decoder_outputs[t] = decoder_output
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
            decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            # all_decoder_outputs[t] = decoder_output
            decoder_input = torch.argmax(decoder_output, dim=1)

    # Loss calculation and backpropagation
    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq * class_szie
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    ) / grad_acc_steps
    
    #tr_loss.backward(retain_graph=True)
    #loss.backward()
    
    #return_loss = loss.item()
    return_loss = loss + tr_loss
    return_loss.backward()

    # Clip gradient norms
    # if clip:
    if use_clip:
        #if not pipeline:
        torch.nn.utils.clip_grad_norm_(stage1_encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        #if not pipeline:
        stage1_encoder_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item(), tr_loss.item(), tr_cor, tr_acc, tr_label_l



def evaluate_lm2seq_v2(input_seq, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, num_list, copy_nums, generate_nums, 
                       stage1_encoder, encoder, decoder, output_lang,
                       beam_size=1, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch).unsqueeze(0)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    stage1_encoder.eval()
    encoder.eval()
    decoder.eval()
    
    #Stage1 span classification
    stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)

    # compute training accuracy
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    tr_logits = stage1_outputs.logits
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    #extend stage1 prediction to word unit length
    #extended_predictions = extend_prediction_span(input_var, predictions, stage1_span_length)
    
    tr_acc = 0
    if labels.tolist() == predictions.tolist():
        tr_acc+=1

    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    
    extended_prediction = []
    for k,v in zip(predictions.tolist(), sentence_length_batch):
        extended_prediction.extend([k] * v)
    extended_predictions_var = torch.LongTensor(extended_prediction).unsqueeze(0)
    assert extended_predictions_var.shape == input_var.shape
    if USE_CUDA:
        extended_predictions_var = extended_predictions_var.cuda()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, stage1_span_ids=extended_predictions_var)
    encoder_outputs = encoder_outputs.transpose(0,1)
    encoder_hidden = encoder_hidden.unsqueeze(0)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]])  # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    if beam_search:
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

        # Run through decoder
        for di in range(max_length):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == output_lang.word2index["[EOS]"]:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0].all_output
            beam_scores = torch.zeros(decoder.output_size * beam_len)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == output_lang.word2index["[EOS]"]:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
                #                                1, num_start, copy_nums, generate_nums, english, var_nums=var_nums)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_input = decoder_input.unsqueeze(0)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
                score = f.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(beam_size)

            for k in range(beam_size):
                word_n = int(topi[k])
                word_input = word_n % decoder.output_size
                temp_input = torch.LongTensor([word_input])
                indices = int(word_n / decoder.output_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices]+[word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:beam_size]
        all_outputs = beam_list[0].all_output
    else:
        all_outputs = []
        for di in range(max_length):
            # if batch_size == 1:
            #     rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            # else:
            #     rule_mask = generate_rule_mask(decoder_input, num_list, output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_input = decoder_input.unsqueeze(0)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            decoder_output = f.log_softmax(decoder_output, dim=1)
            # all_outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=1)
            all_outputs.append(decoder_input)
        all_outputs = torch.stack(all_outputs, dim=1)  # B x S x N
        all_outputs = all_outputs.cpu().detach().numpy()[0]

    return all_outputs, tr_cor, tr_acc, tr_label_l



def train_graph2seq(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
                    generate_nums, batch_graph, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang,
                    use_clip=False, clip=0,
                    use_teacher_forcing=1, scheduled_sampling=False, beam_size=1, beam_search=True,
                    grad_acc=False, zero_grad=True, grad_acc_steps=1, var_nums=[], english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["[UNK]"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计

    batch_graph = torch.LongTensor(batch_graph)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        batch_graph = batch_graph.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, batch_graph)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]] * batch_size)

    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if scheduled_sampling:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            if random.random() < use_teacher_forcing:
                decoder_input = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                target[t] = decoder_input
            else:
                target[t] = generate_decoder_input(
                    target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
                decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
                decoder_input = torch.argmax(decoder_output, dim=1)

    elif random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            else:
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, copy.deepcopy(nums_stack_batch), num_start, unk)
            target[t] = decoder_input
    elif beam_search:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                #                                num_start, copy_nums, generate_nums, english,var_nums=var_nums)
                decoder_input = decoder_input.unsqueeze(0)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) #+ rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output

            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
    else:
        # greedy search
        for t in range(max_target_length):
            # rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
            #                                num_start, copy_nums, generate_nums, english,var_nums=var_nums)
            decoder_input = decoder_input.unsqueeze(0)

            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)

            all_decoder_outputs[t] = decoder_output
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], copy.deepcopy(nums_stack_batch), num_start, unk)
            decoder_output = f.log_softmax(decoder_output, dim=1) #+ rule_mask  # B x classes_size
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            # all_decoder_outputs[t] = decoder_output
            decoder_input = torch.argmax(decoder_output, dim=1)


    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq * class_szie
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    ) / grad_acc_steps

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    # if clip:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        decoder_optimizer.step()

    return return_loss


def evaluate_graph2seq(input_seq, input_length, num_list, copy_nums, generate_nums, batch_graph, encoder, decoder, output_lang,
                     beam_size=1, beam_search=True, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        batch_graph = batch_graph.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], batch_graph)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["[SOS]"]])  # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder # raw code
    decoder_hidden = encoder_hidden[-decoder.n_layers:]
    if beam_search:
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

        # Run through decoder
        for di in range(max_length):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == output_lang.word2index["[EOS]"]:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0].all_output
            beam_scores = torch.zeros(decoder.output_size * beam_len)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == output_lang.word2index["[EOS]"]:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
                #                                1, num_start, copy_nums, generate_nums, english,var_nums=var_nums)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()
                else:
                    # rule_mask = rule_mask.clone()
                    decoder_input = decoder_input.clone()

                decoder_input = decoder_input.unsqueeze(0)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
                score = f.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(beam_size)

            for k in range(beam_size):
                word_n = int(topi[k])
                word_input = word_n % decoder.output_size
                temp_input = torch.LongTensor([word_input])
                indices = int(word_n / decoder.output_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices]+[word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:beam_size]
        all_outputs = beam_list[0].all_output
    else:
        all_outputs = []
        for di in range(max_length):
            # if batch_size == 1:
            #     rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            # else:
            #     rule_mask = generate_rule_mask(decoder_input, num_list, output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            else:
                # rule_mask = rule_mask.clone()
                decoder_input = decoder_input.clone()

            decoder_input = decoder_input.unsqueeze(0)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            decoder_output = f.log_softmax(decoder_output, dim=1)
            # all_outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=1)
            all_outputs.append(decoder_input)
        all_outputs = torch.stack(all_outputs, dim=1)  # B x S x N
        all_outputs = all_outputs.cpu().detach().numpy()[0]

    return all_outputs


def train_seq2tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                   encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
                   merge_optimizer, output_lang, num_pos, var_nums=[], use_clip=False, clip=0.0,
                   grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums) # 最大的位置列表数目+常识数字数目+未知数列表
    for i in num_size_batch:
        d = i + len(generate_nums) + len(var_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    # print(max_num_size)
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :] # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)] # root embedding B x 1

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start - len(var_nums)
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
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

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_seq2tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
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

        return beams[0].out
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
        return all_node_outputs[0]


def train_lm2tree_bystep(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, 
                  nums_stack_batch, num_size_batch, stage1_span_ids_batch, stage1_sentence_ids_batch, 
                  quantity_indicator_batch, cls_loc_batch, attention_mask_sentence_batch, equation_id_batch, 
                  var_cnt_batch, stage1_span_ids2_batch, target2_batch, target2_length, nums_stack2_batch, cls_loc2_batch,
                  attention_mask_sentence2_batch, equation_id2_batch, generate_nums, encoder, predict, generate, merge, 
                  eq1_encoder, encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, 
                  eq1_encoder_optimizer, output_lang, num_pos, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
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
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    stage1_span_ids2_var = torch.LongTensor(stage1_span_ids2_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch)
    
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_span_ids_var = stage1_span_ids_var[:, :max_len]
    stage1_span_ids2_var = stage1_span_ids2_var[:, :max_len]
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    quantity_indicator_var = quantity_indicator_var[:, :max_len]
    
    equation_id_var = torch.LongTensor(equation_id_batch)
    equation_id2_var = torch.LongTensor(equation_id2_batch)
    #equation_id_var = equation_id_var.unsqueeze(-1).expand_as(input_var)
    var_cnt_tensor = torch.LongTensor(var_cnt_batch)-1
    #var_cnt_tensor = var_cnt_tensor.unsqueeze(-1).expand_as(input_var)

    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch)
    cls_loc_var = torch.LongTensor(cls_loc_batch)
    attention_mask_sentence2_var = torch.BoolTensor(attention_mask_sentence2_batch)
    cls_loc2_var = torch.LongTensor(cls_loc2_batch)
    max_stage1_len = 9
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    cls_loc_var = cls_loc_var[:, :max_stage1_len]
    attention_mask_sentence2_var = attention_mask_sentence2_var[:, :max_stage1_len]
    cls_loc2_var = cls_loc2_var[:, :max_stage1_len]
    
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计
    target2 = torch.LongTensor(target2_batch).transpose(0, 1)
    max_target2_len = max(target2_length)
    target2 = target2[:max_target2_len, :]  # 因为梯度累计
#     print(id_batch)
#     print(target)
#     print(target2)
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    padding_hidden2 = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    eq1_encoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        stage1_span_ids2_var = stage1_span_ids2_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        padding_hidden2 = padding_hidden2.cuda()
        num_mask = num_mask.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        attention_mask_sentence2_var = attention_mask_sentence2_var.cuda()
        cls_loc_var = cls_loc_var.cuda()
        cls_loc2_var = cls_loc2_var.cuda()
        equation_id_var = equation_id_var.cuda()
        equation_id2_var = equation_id2_var.cuda()
        var_cnt_tensor = var_cnt_tensor.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
        eq1_encoder_optimizer.zero_grad()
    
    # Run words through encoder-1
    for b in range(batch_size):
        #print("one")
        encoder_outputs, problem_output = encoder(input_batch=input_var[b].unsqueeze(0), 
                                                  attention_mask=attention_mask_var[b].unsqueeze(0),
                                                  token_type_ids=token_type_ids_var[b].unsqueeze(0), 
                                                  sentence_ids=stage1_sentence_ids_var[b].unsqueeze(0), 
                                                  quantity_ids=quantity_indicator_var[b].unsqueeze(0), 
                                                  stage1_ids=stage1_span_ids_var[b].unsqueeze(0),
                                                  cls_loc=cls_loc_var[b].unsqueeze(0), 
                                                  attention_mask2=attention_mask_sentence_var[b].unsqueeze(0),
                                                  equation_index_ids=equation_id_var[b].unsqueeze(0), 
                                                  var_cnt_ids=var_cnt_tensor[b].unsqueeze(0))
        encoder_outputs = encoder_outputs.transpose(0,1)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

        max_target_length = max(target_length)

        all_node_outputs = []
        all_target_node = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos[b]], 1, num_size,
                                                                  encoder.config.hidden_size)

        num_start = output_lang.num_start - len(var_nums)
        embeddings_stacks = [[] for _ in range(1)]
        left_childs = [None for _ in range(1)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask[b].unsqueeze(0), num_mask[b].unsqueeze(0))

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_target_node.append(current_embeddings.squeeze(0))
            all_node_outputs.append(outputs)
            
            target_t, generate_input = generate_tree_input(target[t][b].unsqueeze(0).tolist(), outputs, copy.deepcopy(nums_stack_batch[b]), num_start, unk)
            target[t][b] = target_t
            if USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(1), left_child.split(1), right_child.split(1),
                                                   node_stacks, target[t][b].unsqueeze(0).tolist(), embeddings_stacks):
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

        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        all_target_node = torch.stack(all_target_node, dim=1)
        target_conti = target.transpose(0, 1).contiguous()
        if USE_CUDA:
            all_node_outputs = all_node_outputs.cuda()
            all_target_node = all_target_node.cuda()
            target_conti = target_conti.cuda()
        if b == 0:
            loss = masked_cross_entropy(all_node_outputs, target_conti[b].unsqueeze(0), [target_length[b]]) / grad_acc_steps
        else:
            loss += masked_cross_entropy(all_node_outputs, target_conti[b].unsqueeze(0), [target_length[b]]) / grad_acc_steps
        #print(loss)
        if var_cnt_tensor[b] == 1:
            #print("two")
            encoder_outputs, problem_output = encoder(input_batch=input_var[b].unsqueeze(0), 
                                                      attention_mask=attention_mask_var[b].unsqueeze(0),
                                                      token_type_ids=token_type_ids_var[b].unsqueeze(0), 
                                                      sentence_ids=stage1_sentence_ids_var[b].unsqueeze(0), 
                                                      quantity_ids=quantity_indicator_var[b].unsqueeze(0), 
                                                      stage1_ids=stage1_span_ids2_var[b].unsqueeze(0),
                                                      cls_loc=cls_loc2_var[b].unsqueeze(0), 
                                                      attention_mask2=attention_mask_sentence2_var[b].unsqueeze(0),
                                                      equation_index_ids=equation_id2_var[b].unsqueeze(0), 
                                                      var_cnt_ids=var_cnt_tensor[b].unsqueeze(0))
            encoder_outputs = encoder_outputs.transpose(0,1)
            #padded_hidden = sequence_mask(sequence_length=target_length, max_len=max_target_length)
            problem_output = eq1_encoder(problem_output, all_target_node, [target_length[b]])
            # Prepare input and output variables
            node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1
            max_target2_length = max(target2_length)

            all_node_outputs2 = []
            copy_num_len = [len(_) for _ in num_pos]
            num_size = max(copy_num_len)
            all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos[b]], 1, num_size,
                                                                      encoder.config.hidden_size)

            num_start = output_lang.num_start - len(var_nums)
            embeddings_stacks = [[] for _ in range(1)]
            left_childs = [None for _ in range(1)]
            for t in range(max_target2_length):
                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden2, seq_mask[b].unsqueeze(0), num_mask[b].unsqueeze(0))

                # all_leafs.append(p_leaf)
                outputs = torch.cat((op, num_score), 1)
                all_node_outputs2.append(outputs)

                target_t, generate_input = generate_tree_input(target2[t][b].unsqueeze(0).tolist(), outputs, copy.deepcopy(nums_stack2_batch[b]), num_start, unk)
                target2[t][b] = target_t
                if USE_CUDA:
                    generate_input = generate_input.cuda()
                left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(1), left_child.split(1), right_child.split(1),
                                                       node_stacks, target2[t][b].unsqueeze(0).tolist(), embeddings_stacks):
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

            all_node_outputs2 = torch.stack(all_node_outputs2, dim=1)  # B x S x N
            target2_conti = target2.transpose(0, 1).contiguous()
            if USE_CUDA:
                all_node_outputs2 = all_node_outputs2.cuda()
                target2_conti = target2_conti.cuda()
            #print(masked_cross_entropy(all_node_outputs2, target2_conti[b].unsqueeze(0), [target2_length[b]]) / grad_acc_steps)
            loss += masked_cross_entropy(all_node_outputs2, target2_conti[b].unsqueeze(0), [target2_length[b]]) / grad_acc_steps
    
    loss.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(eq1_encoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
        eq1_encoder_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_lm2tree_bystep(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, stage1_sentence_ids_batch, quantity_indicator_batch, cls_loc_batch, attention_mask_sentence_batch, equation_id_batch, var_cnt_batch, stage1_span_ids2_batch, cls_loc2_batch, attention_mask_sentence2_batch, equation_id2_batch, generate_nums, encoder, predict, generate, merge, eq1_encoder, output_lang, num_pos, beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    stage1_span_ids2_var = torch.LongTensor(stage1_span_ids2_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch).unsqueeze(0)
    attention_mask_sentence2_var = torch.BoolTensor(attention_mask_sentence2_batch).unsqueeze(0)
    cls_loc_var = torch.LongTensor(cls_loc_batch).unsqueeze(0)
    cls_loc2_var = torch.LongTensor(cls_loc2_batch).unsqueeze(0)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)
    var_cnt_tensor = torch.LongTensor([var_cnt_batch])-1
    equation_id_var = torch.LongTensor([equation_id_batch])
    equation_id2_var = torch.LongTensor([equation_id2_batch])
#     var_cnt_tensor = (torch.LongTensor([var_cnt_batch])-1).unsqueeze(0).expand_as(input_var)
#     equation_id_var = torch.LongTensor([equation_id_batch]).unsqueeze(0).expand_as(input_var)
    
    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    #eq1_encoder.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1
    total_beam_res = []

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        stage1_span_ids2_var = stage1_span_ids2_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        attention_mask_sentence2_var = attention_mask_sentence2_var.cuda()
        cls_loc_var = cls_loc_var.cuda()
        cls_loc2_var = cls_loc2_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        var_cnt_tensor = var_cnt_tensor.cuda()
        equation_id_var = equation_id_var.cuda()
        equation_id2_var = equation_id2_var.cuda()
    # Run words through encoder
    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    encoder_outputs, problem_output = encoder(input_batch=input_var, 
                                              attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, 
                                              sentence_ids=stage1_sentence_ids_var, 
                                              quantity_ids=quantity_indicator_var, 
                                              stage1_ids=stage1_span_ids_var,
                                              cls_loc=cls_loc_var, 
                                              attention_mask2=attention_mask_sentence_var,
                                              equation_index_ids=equation_id_var, 
                                              var_cnt_ids=var_cnt_tensor)
    encoder_outputs = encoder_outputs.transpose(0,1)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
    num_start = output_lang.num_start - len(var_nums)
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    if beam_search:
        #all_node_outputs = []
        #all_target_node = []
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
        #beams2 = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)
                current_target_node = copy_list(b.target_node)
                current_target_node.append(current_embeddings.squeeze(0))
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                topv, topi = out_score.topk(beam_size)

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
                                                  current_left_childs, current_out, current_target_node))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            #print(len(all_target_node))
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        #return beams[0].out
        all_target_node = torch.stack(beams[0].target_node, dim=1)  # B x S x N
        if USE_CUDA:
            all_target_node = all_target_node.cuda()
        total_beam_res.append(beams[0].out)
    
        if var_cnt_tensor == 1:
            encoder_outputs, problem_output = encoder(input_batch=input_var, 
                                                  attention_mask=attention_mask_var,
                                                  token_type_ids=token_type_ids_var, 
                                                  sentence_ids=stage1_sentence_ids_var, 
                                                  quantity_ids=quantity_indicator_var, 
                                                  stage1_ids=stage1_span_ids2_var,
                                                  cls_loc=cls_loc2_var, 
                                                  attention_mask2=attention_mask_sentence2_var,
                                                  equation_index_ids=equation_id2_var, 
                                                  var_cnt_ids=var_cnt_tensor)
            encoder_outputs = encoder_outputs.transpose(0,1)
            #padded_hidden = sequence_mask(sequence_length=len(all_node_outputs), max_len=len(all_node_outputs))
            #problem_output = eq1_encoder(problem_output, all_target_node, [all_target_node.size(1)])
            node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

            num_size = len(num_pos)
            all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                                      encoder.config.hidden_size)
            num_start = output_lang.num_start - len(var_nums)
            embeddings_stacks = [[] for _ in range(batch_size)]
            left_childs = [None for _ in range(batch_size)]
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            for t in range(max_length):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs

                    num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                        b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                        seq_mask, num_mask)
                    current_target_node = copy_list(b.target_node)
                    current_target_node.append(current_embeddings.squeeze(0))
                    out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                    topv, topi = out_score.topk(beam_size)

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
                                                      current_left_childs, current_out, current_target_node))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break

            #return beams[0].out
            total_beam_res.append(beams[0].out)
        return total_beam_res
    
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
        
        return all_node_outputs[0]
        
        if var_cnt_tensor == 1:
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

def train_lm2tree(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, 
                  nums_stack_batch, num_size_batch, stage1_span_ids_batch, stage1_sentence_ids_batch, 
                  quantity_indicator_batch, cls_loc_batch, attention_mask_sentence_batch, equation_id_batch, 
                  var_cnt_batch, prev_eq_batch, prev_eq_len_batch, prev_eq_num_pos, generate_nums, 
                  encoder, predict, generate, merge, eq1_encoder,
                  encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, eq1_encoder_optimizer,
                  output_lang, num_pos, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
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
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch)
    prev_eq_num_pos_var = torch.LongTensor(prev_eq_num_pos)
    
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_span_ids_var = stage1_span_ids_var[:, :max_len]
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    quantity_indicator_var = quantity_indicator_var[:, :max_len]
    prev_eq_num_pos_var = prev_eq_num_pos_var[:, :max_len]
    
    equation_id_var = torch.LongTensor(equation_id_batch)
    #equation_id_var = equation_id_var.unsqueeze(-1).expand_as(input_var)
    var_cnt_tensor = torch.LongTensor(var_cnt_batch)-1
    #var_cnt_tensor = var_cnt_tensor.unsqueeze(-1).expand_as(input_var)

    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch)
    cls_loc_var = torch.LongTensor(cls_loc_batch)
    max_stage1_len = 9
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    cls_loc_var = cls_loc_var[:, :max_stage1_len]
    
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计
    prev_eq = torch.LongTensor(prev_eq_batch)
    max_prev_eq_len = max(prev_eq_len_batch)
    if max_prev_eq_len == 0:
        max_prev_eq_len = 1
    #attention_mask_prev_eq = torch.BoolTensor(attention_mask_prev_eq_batch)
    prev_eq = prev_eq[:, :max_prev_eq_len]
    #attention_mask_prev_eq = attention_mask_prev_eq[:, :max_prev_eq_len]

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    #eq1_encoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        cls_loc_var = cls_loc_var.cuda()
        equation_id_var = equation_id_var.cuda()
        var_cnt_tensor = var_cnt_tensor.cuda()
        prev_eq = prev_eq.cuda()
        prev_eq_num_pos_var = prev_eq_num_pos_var.cuda()
        #attention_mask_prev_eq = attention_mask_prev_eq.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
        #eq1_encoder_optimizer.zero_grad()
    
    # Run words through encoder-1
    encoder_outputs, problem_output = encoder(input_batch=input_var, 
                                              attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, 
                                              sentence_ids=stage1_sentence_ids_var, 
                                              quantity_ids=quantity_indicator_var, 
                                              stage1_ids=stage1_span_ids_var,
                                              cls_loc=cls_loc_var, 
                                              attention_mask2=attention_mask_sentence_var,
                                              equation_index_ids=equation_id_var, 
                                              var_cnt_ids=var_cnt_tensor,
                                              output_eq_ids=prev_eq_num_pos_var)
    num_start = output_lang.num_start - len(var_nums)
    #problem_output = eq1_encoder(problem_output, encoder_outputs, prev_eq, num_start, prev_eq_num_pos, prev_eq_len_batch)
    encoder_outputs = encoder_outputs.transpose(0,1)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

    max_target_length = max(target_length)

    all_node_outputs = []
    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    #copy_eq_num_len = [len(_) for _ in prev_eq_num_pos]
    #eq_num_size = max(copy_eq_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.config.hidden_size)
    #all_eq_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, prev_eq_num_pos, batch_size, eq_num_size,
    #                                                          encoder.config.hidden_size
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        #num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
        #    eq1_encoder.embedding_weight, node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, 
        #    padding_hidden, seq_mask, num_mask)
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, 
            padding_hidden, seq_mask, num_mask)
        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        #left_child, right_child, node_label = generate(eq1_encoder.op_embeddings, current_embeddings, generate_input, current_context)
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

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    
    loss.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)
        #torch.nn.utils.clip_grad_norm_(eq1_encoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
        #eq1_encoder_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_lm2tree(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, stage1_sentence_ids_batch, quantity_indicator_batch, cls_loc_batch, attention_mask_sentence_batch, equation_id_batch, var_cnt_batch, prev_eq_batch, prev_eq_length, prev_eq_num_pos, generate_nums, encoder, predict, generate, merge, eq1_encoder, output_lang, num_pos, beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch).unsqueeze(0)
    cls_loc_var = torch.LongTensor(cls_loc_batch).unsqueeze(0)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)
    var_cnt_tensor = torch.LongTensor([var_cnt_batch])-1
    equation_id_var = torch.LongTensor([equation_id_batch])
    prev_eq = torch.LongTensor(prev_eq_batch).unsqueeze(0)
    prev_eq_num_pos_var = torch.LongTensor(prev_eq_num_pos).unsqueeze(0)
#     var_cnt_tensor = (torch.LongTensor([var_cnt_batch])-1).unsqueeze(0).expand_as(input_var)
#     equation_id_var = torch.LongTensor([equation_id_batch]).unsqueeze(0).expand_as(input_var)
    
    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    #eq1_encoder.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        cls_loc_var = cls_loc_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        var_cnt_tensor = var_cnt_tensor.cuda()
        equation_id_var = equation_id_var.cuda()
        prev_eq = prev_eq.cuda()
        prev_eq_num_pos_var = prev_eq_num_pos_var.cuda()
    # Run words through encoder
    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    encoder_outputs, problem_output = encoder(input_batch=input_var, 
                                              attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, 
                                              sentence_ids=stage1_sentence_ids_var, 
                                              quantity_ids=quantity_indicator_var, 
                                              stage1_ids=stage1_span_ids_var,
                                              cls_loc=cls_loc_var, 
                                              attention_mask2=attention_mask_sentence_var,
                                              equation_index_ids=equation_id_var, 
                                              var_cnt_ids=var_cnt_tensor,
                                              output_eq_ids=prev_eq_num_pos_var)
    num_start = output_lang.num_start - len(var_nums)
    #problem_output = eq1_encoder(problem_output, encoder_outputs, prev_eq, num_start, [prev_eq_num_pos], [prev_eq_length])
    encoder_outputs = encoder_outputs.transpose(0,1)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
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
                left_childs = b.left_childs

                #num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                #    eq1_encoder.embedding_weight, b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, 
                #    padding_hidden, seq_mask, num_mask)
                
                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, 
                    padding_hidden, seq_mask, num_mask)
                
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                topv, topi = out_score.topk(beam_size)

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
                        #left_child, right_child, node_label = generate(eq1_encoder.op_embeddings, current_embeddings, 
                        #                                               generate_input, current_context)
                        left_child, right_child, node_label = generate(current_embeddings, 
                                                                       generate_input, current_context)

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

        return beams[0].out
    
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
        
        return all_node_outputs[0]    
    
def train_stage1_v1(input_batch, input_length, attention_mask_batch, token_type_ids_batch,  
                  stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                  attention_mask_sentence_batch, sentence_length_batch, quantity_indicator_batch, sep_loc_batch,
                  stage1_encoder, stage1_encoder_optimizer, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5):
    #print(id_batch)
    max_len = max(input_length)

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch)
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    quantity_indicator_var = quantity_indicator_var[:, :max_len]

    #stage1 label
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    #attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch)
    sep_loc_var = torch.LongTensor(sep_loc_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    sep_loc_var = sep_loc_var[:, :max_stage1_len]
    stage1_span_length_var = torch.LongTensor(stage1_span_length)
    #var_cnt_tensor = torch.LongTensor(var_cnt_batch)
    stage1_encoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        sep_loc_var = sep_loc_var.cuda()
        stage1_span_length_var = stage1_span_length_var.cuda()
        #var_cnt_tensor = var_cnt_tensor.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        stage1_encoder_optimizer.zero_grad()
    
    #Stage1 span classification
    #print(stage1_span_ids_var)
    #tr_logits, tr_loss = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_indicator_var, stage1_span_ids_var, sentence_length_batch, stage1_span_length_var, sep_loc_var, var_cnt_tensor)
    tr_logits, tr_loss = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_indicator_var, stage1_span_ids_var, sentence_length_batch, stage1_span_length_var, sep_loc_var)

    # compute training accuracy
    #tr_loss = stage1_outputs.loss
    #tr_logits = stage1_outputs.logits
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    #print(active_logits.shape)
   
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
    flattened_targets = stage1_span_ids_var.view(-1)
    
    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    

    l = 0
    tmp_acc = 0
    for ixx, i in enumerate(stage1_span_length):
        label_e = labels.tolist()[l:(l+i)]
        prediction_e = predictions.tolist()[l:(l+i)]
        if label_e == prediction_e:
            tmp_acc+=1
#         else:
#             logs_content = "{}".format(index[ixx])
#             add_log(log_file, logs_content)
#             logs_content = "Label: {}".format(label_e)
#             add_log(log_file, logs_content)
#             logs_content = "Prediction: {}".format(prediction_e)
#             add_log(log_file, logs_content)
        l+=i

    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    tr_acc = tmp_acc
    loss = tr_loss / grad_acc_steps
    loss.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(stage1_encoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        stage1_encoder_optimizer.step()
    
    return loss.item(), tr_cor, tr_acc, tr_label_l  # , loss_0.item(), loss_1.item()
    
    
    
def evaluate_stage1_v1(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, quantity_indicator_batch, sep_loc_batch, stage1_encoder,
                       beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch).unsqueeze(0)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    sep_loc_var = torch.LongTensor(sep_loc_batch).unsqueeze(0)
    #stage1_span_length_var = torch.LongTensor([len(stage1_span_ids_batch)]).unsqueeze(0)
    # Set to not-training mode to disable dropout
    stage1_encoder.eval()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        sep_loc_var = sep_loc_var.cuda()
        #stage1_span_length_var = stage1_span_length_var.cuda()
    
    #Stage1 span classification
    tr_logits, tr_loss = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_indicator_var, stage1_span_ids_var, [sentence_length_batch], [stage1_span_length], sep_loc_var)

    # compute training accuracy
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    #tr_logits = stage1_outputs.logits
    #tr_loss = stage1_outputs.loss
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
    #print(labels)
    #print(predictions)
    tmp_acc = 0
    if labels.tolist() == predictions.tolist():
        tmp_acc+=1
    #else:
        #logs_content = "{}".format(next(i for i in data if i["iIndex"] == index[ixx].item()))
        #add_log(log_file, logs_content)
        #logs_content = "Label: {}".format(labels.tolist())
        #add_log(log_file, logs_content)
        #logs_content = "Prediction: {}".format(predictions.tolist())
        #add_log(log_file, logs_content)

    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    tr_acc = tmp_acc
    
    return tr_loss.item(), tr_cor, tr_acc, tr_label_l, predictions.tolist()
    

    
    
def train_stage1_v2(input_batch, input_length, attention_mask_batch, token_type_ids_batch,  
                  stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                  attention_mask_sentence_batch, sentence_length_batch, quantity_indicator_batch, sep_loc_batch,
                  var_cnt_batch, stage1_encoder, stage1_encoder_optimizer, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5):
    #print(id_batch)
    max_len = max(input_length)

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch)
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    quantity_indicator_var = quantity_indicator_var[:, :max_len]

    #stage1 label
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    #attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch)
    sep_loc_var = torch.LongTensor(sep_loc_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    sep_loc_var = sep_loc_var[:, :max_stage1_len]
    stage1_span_length_var = torch.LongTensor(stage1_span_length)
    var_cnt_tensor = torch.LongTensor(var_cnt_batch)-1
    
    stage1_encoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        sep_loc_var = sep_loc_var.cuda()
        stage1_span_length_var = stage1_span_length_var.cuda()
        var_cnt_tensor = var_cnt_tensor.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        stage1_encoder_optimizer.zero_grad()
    
    #Stage1 span classification
    #print(stage1_span_ids_var)
    tr_logits, tr_loss, var_cnt_pred, var_cnt_loss = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_indicator_var, stage1_span_ids_var, sentence_length_batch, stage1_span_length_var, sep_loc_var, var_cnt_tensor)

    # compute training accuracy
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
    flattened_targets = stage1_span_ids_var.view(-1)
    
    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
    #compute var cnt accuracy
    var_cnt_pred = var_cnt_pred.view(-1, 2) # shape (batch_size * seq_len, num_labels)
    var_cnt_preds = torch.argmax(var_cnt_pred, axis=1) # shape (batch_size * seq_len,)
    #print(var_cnt_preds)
    var_cnt_acc = 0
    for i, j in zip(var_cnt_preds, var_cnt_tensor):
        if i==j:
            var_cnt_acc+=1
    
    l = 0
    tmp_acc = 0
    for ixx, i in enumerate(stage1_span_length):
        label_e = labels.tolist()[l:(l+i)]
        prediction_e = predictions.tolist()[l:(l+i)]
        if label_e == prediction_e:
            tmp_acc+=1
        l+=i

    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    tr_acc = tmp_acc
    loss = tr_loss / grad_acc_steps
    var_cnt_loss1 = var_cnt_loss / grad_acc_steps
    total_loss = loss + var_cnt_loss1
    total_loss.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(stage1_encoder.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        stage1_encoder_optimizer.step()
    
    return total_loss.item(), tr_cor, tr_acc, tr_label_l, var_cnt_acc  # , loss_0.item(), loss_1.item()
    
    
    
def evaluate_stage1_v2(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, quantity_indicator_batch, sep_loc_batch, var_cnt_batch, stage1_encoder,
                       beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.BoolTensor(attention_mask_sentence_batch).unsqueeze(0)
    quantity_indicator_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    sep_loc_var = torch.LongTensor(sep_loc_batch).unsqueeze(0)
    var_cnt_tensor = torch.LongTensor([var_cnt_batch])-1
    # Set to not-training mode to disable dropout
    stage1_encoder.eval()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        quantity_indicator_var = quantity_indicator_var.cuda()
        sep_loc_var = sep_loc_var.cuda()
        var_cnt_tensor = var_cnt_tensor.cuda()
    
    #Stage1 span classification
    tr_logits, tr_loss, var_cnt_pred, var_cnt_loss = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_indicator_var, stage1_span_ids_var, [sentence_length_batch], [stage1_span_length], sep_loc_var, var_cnt_tensor)

    # compute training accuracy
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
    #compute var cnt accuracy
    var_cnt_pred = var_cnt_pred.view(-1, 2) # shape (batch_size * seq_len, num_labels)
    var_cnt_preds = torch.argmax(var_cnt_pred, axis=1) # shape (batch_size * seq_len,)
    #print(var_cnt_preds)
    var_cnt_acc = 0
    for i, j in zip(var_cnt_preds, var_cnt_tensor):
        if i==j:
            var_cnt_acc+=1
            
    tmp_acc = 0
    if labels.tolist() == predictions.tolist():
        tmp_acc+=1
    #else:
        #logs_content = "{}".format(next(i for i in data if i["iIndex"] == index[ixx].item()))
        #add_log(log_file, logs_content)
        #logs_content = "Label: {}".format(labels.tolist())
        #add_log(log_file, logs_content)
        #logs_content = "Prediction: {}".format(predictions.tolist())
        #add_log(log_file, logs_content)
    
    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    tr_acc = tmp_acc
    total_loss = tr_loss + var_cnt_loss
    
    return total_loss.item(), tr_cor, tr_acc, tr_label_l, var_cnt_acc, predictions.tolist()
    
    
    
def train_lm2tree_v2(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, 
                  nums_stack_batch, num_size_batch, stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                  attention_mask_sentence_batch, sentence_length_batch, generate_nums,
                  stage1_encoder, encoder, predict, generate, merge, 
                  stage1_encoder_optimizer, encoder_optimizer, predict_optimizer, generate_optimizer,
                  merge_optimizer, output_lang, num_pos, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5):
    #print(id_batch)
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    #print(max_len)
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
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)
    #print(input_batch)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch)
    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计
    stage1_sentence_ids_var = stage1_sentence_ids_var[:, :max_len]
    
    
    #stage1 label
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch)
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    
    #print(stage1_span_ids_var)
    #print(attention_mask_sentence_var)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    stage1_encoder.train()
    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        stage1_sentence_ids_var = stage1_sentence_ids_var.cuda()
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        stage1_encoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
    
    #Stage1 span classification
    #sep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)
    #nosep version
    stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var, sentence_length_batch)

    # compute training accuracy
    tr_loss = stage1_outputs.loss
    tr_logits = stage1_outputs.logits
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    

    #extend stage1 prediction to word unit length
    #extended_predictions = extend_prediction_span(input_var, predictions, stage1_span_length)
    extended_predictions = []
    l = 0
    tr_acc = 0
    for i,j in enumerate(stage1_span_length):
        label_e = labels.tolist()[l:(l+j)]
        prediction_e = predictions.tolist()[l:(l+j)]
        l+=j
        assert j == len(sentence_length_batch[i])
        if label_e == prediction_e:
            tr_acc+=1
        extended_prediction = []
        #print(prediction_e)
        for k,v in zip(prediction_e, sentence_length_batch[i]):
            extended_prediction.extend([k] * v)
        extended_prediction.extend([0] * (max_len-len(extended_prediction)))
        extended_predictions.append(extended_prediction)
    
    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())

    #print(input_batch)
    #print(extended_predictions)
    extended_predictions_var = torch.LongTensor(extended_predictions)
    extended_predictions_var = extended_predictions_var[:, :max_len]
    assert extended_predictions_var.shape == input_var.shape
    if USE_CUDA:
        extended_predictions_var = extended_predictions_var.cuda()
    
    # Run words through encoder
    encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, stage1_ids=extended_predictions_var)
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

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
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

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss_eq = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    loss = loss_eq + tr_loss
    #tr_loss.backward(retain_graph=True)
    loss.backward()
    #loss_eq.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(stage1_encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        stage1_encoder_optimizer.step()
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
    
    #print(tr_loss)
    return loss_eq.item(), tr_loss.item(), tr_cor, tr_acc, tr_label_l  # , loss_0.item(), loss_1.item()
    
    
def evaluate_lm2tree_v2(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, generate_nums, 
                       stage1_encoder, encoder, predict, generate, merge, output_lang, num_pos,
                       beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):
    #print(stage1_span_ids_batch)
    #print(input_batch)
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch).unsqueeze(0)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    stage1_encoder.eval()
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
        stage1_span_ids_var = stage1_span_ids_var.cuda()
        attention_mask_sentence_var = attention_mask_sentence_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    
    #Stage1 span classification
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)
    stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var, [sentence_length_batch])
    

    # compute training accuracy
    #flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    tr_logits = stage1_outputs.logits
    #tr_loss = stage1_outputs.loss
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
    
    extended_prediction = []
    for k,v in zip(predictions.tolist(), sentence_length_batch):
        extended_prediction.extend([k] * v)
    extended_predictions_var = torch.LongTensor(extended_prediction).unsqueeze(0)
    assert extended_predictions_var.shape == input_var.shape
    if USE_CUDA:
        extended_predictions_var = extended_predictions_var.cuda()
    
    # Run words through encoder
    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, stage1_ids=extended_predictions_var)
    encoder_outputs = encoder_outputs.transpose(0,1)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
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
    
#gumbel + quantity indicator
def train_lm2tree_v3(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, 
                  nums_stack_batch, num_size_batch, stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                  attention_mask_sentence_batch, sentence_length_batch, quantity_indicator_batch, generate_nums,
                  stage1_encoder, encoder, predict, generate, merge, 
                  stage1_encoder_optimizer, encoder_optimizer, predict_optimizer, generate_optimizer,
                  merge_optimizer, output_lang, num_pos, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5):
    #print(id_batch)
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    #print(max_len)
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
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)
    #print(input_batch)
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
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    
    #print(stage1_span_ids_var)
    #print(attention_mask_sentence_var)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    stage1_encoder.train()
    encoder.train()
    predict.train()
    generate.train()
    merge.train()

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

    # Zero gradients of both optimizers
    if zero_grad:
        stage1_encoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
    
    #Stage1 span classification
    #sep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)
    #nosep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var, sentence_length_batch)
    tr_logits, tr_loss, tr_pred = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_ids_var, stage1_span_ids_var, sentence_length_batch)
    #print(tr_pred.shape)
    # compute training accuracy
    #tr_loss = stage1_outputs.loss
    #tr_logits = stage1_outputs.logits
    #active_logits = tr_logits
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    

    #extend stage1 prediction to word unit length
    #extended_predictions = extend_prediction_span(input_var, predictions, stage1_span_length)
    extended_predictions = []
    l = 0
    tr_acc = 0
    #print(stage1_span_length)
    for i,j in enumerate(stage1_span_length):
        label_e = labels.tolist()[l:(l+j)]
        prediction_e = predictions.tolist()[l:(l+j)]
        l+=j
        assert j == len(sentence_length_batch[i])
        if label_e == prediction_e:
            tr_acc+=1
        extended_prediction = []
        #if max_len > sum(sentence_length_batch[i]):
        #    sentence_length_batch[i].append(max_len-sum(sentence_length_batch[i]))
        #extended_prediction = torch.repeat_interleave(tr_pred[i], torch.tensor(sentence_length_batch[i], device=tr_pred.device), dim=0)
        #extended_prediction = torch.repeat_interleave(tr_pred[i, :j, :], torch.tensor(sentence_length_batch[i], device=tr_pred.device), dim=0)
        #padder = torch.zeros((max_len-extended_prediction.size(0), extended_prediction.size(1)), device=tr_pred.device)
        #padded_extended_prediction = torch.cat([extended_prediction, padder], dim=0)
        for k,v in zip(prediction_e, sentence_length_batch[i]):
            extended_prediction.extend([k] * v)
        extended_prediction.extend([0] * (max_len-len(extended_prediction)))
        extended_predictions.append(extended_prediction)
        
        #extended_predictions.append(padded_extended_prediction)
    
    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    
    #print(input_batch)
    #print(extended_predictions)
    #extended_predictions_var = torch.stack(extended_predictions)
    #print(extended_predictions_var.shape)
    extended_predictions_var = torch.LongTensor(extended_predictions)
    extended_predictions_var = extended_predictions_var[:, :max_len]
    #print(extended_predictions_var.shape)
    #print(input_var.shape)
    assert extended_predictions_var.shape == input_var.shape
    if USE_CUDA:
        extended_predictions_var = extended_predictions_var.cuda()
    
    # Run words through encoder
    encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, sentence_ids=stage1_sentence_ids_var,
                                              stage1_ids=extended_predictions_var)
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

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
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

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss_eq = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    loss = loss_eq + tr_loss
    #tr_loss.backward(retain_graph=True)
    loss.backward()
    #loss_eq.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(stage1_encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        stage1_encoder_optimizer.step()
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
    
    #print(tr_loss)
    return loss_eq.item(), tr_loss.item(), tr_cor, tr_acc, tr_label_l  # , loss_0.item(), loss_1.item()
    
    
def evaluate_lm2tree_v3(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, quantity_indicator_batch, generate_nums, 
                       stage1_encoder, encoder, predict, generate, merge, output_lang, num_pos,
                       beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):
    #print(stage1_span_ids_batch)
    #print(input_batch)
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    quantity_ids_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch).unsqueeze(0)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    stage1_encoder.eval()
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
    
    #Stage1 span classification
    #sep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)
    #nosep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var, [sentence_length_batch])
    
    tr_logits, tr_loss, tr_pred = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_ids_var, stage1_span_ids_var, [sentence_length_batch])
    

    # compute training accuracy
    #flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    #tr_logits = stage1_outputs.logits
    #tr_loss = stage1_outputs.loss
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
    
    extended_prediction = []
    for k,v in zip(predictions.tolist(), sentence_length_batch):
        extended_prediction.extend([k] * v)
    #print(tr_pred.shape)
    #print(sentence_length_batch)
    #extended_prediction = torch.repeat_interleave(tr_pred.squeeze(0), torch.tensor(sentence_length_batch, device=tr_pred.device), dim=0)
    extended_predictions_var = torch.LongTensor(extended_prediction).unsqueeze(0)
    #extended_predictions_var = extended_prediction.unsqueeze(0)
    #print(extended_predictions_var.shape)
    #print(input_var.shape)
    assert extended_predictions_var.shape == input_var.shape
    if USE_CUDA:
        extended_predictions_var = extended_predictions_var.cuda()
    
    # Run words through encoder
    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, sentence_ids=stage1_sentence_ids_var,
                                              stage1_ids=extended_predictions_var)
    encoder_outputs = encoder_outputs.transpose(0,1)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
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
    
#gumbel + quantity indicator + end2end
def train_lm2tree_v4(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, 
                  nums_stack_batch, num_size_batch, stage1_span_ids_batch, stage1_span_length, stage1_sentence_ids_batch, 
                  attention_mask_sentence_batch, sentence_length_batch, quantity_indicator_batch, generate_nums,
                  encoder, predict, generate, merge, 
                  encoder_optimizer, predict_optimizer, generate_optimizer,
                  merge_optimizer, output_lang, num_pos, id_batch, var_nums=[], use_clip=False, clip=0.0,
                  grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False, labels_num=5):
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
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch)
    max_stage1_len = max(stage1_span_length)
    stage1_span_ids_var = stage1_span_ids_var[:, :max_stage1_len]  # 因为梯度累计
    attention_mask_sentence_var = attention_mask_sentence_var[:, :max_stage1_len]
    
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

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

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
    
    #Stage1 span classification
    tr_logits, tr_loss, encoder_outputs, problem_output = encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_ids_var, stage1_span_ids_var, sentence_length_batch, stage1_span_length)

    # compute training accuracy
    active_logits = tr_logits.view(-1, labels_num) # shape (batch_size * seq_len, num_labels)
    flattened_targets = stage1_span_ids_var.view(-1) # shape (batch_size * seq_len,)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

    # only compute accuracy at active labels
    active_accuracy = stage1_span_ids_var.view(-1) != -100.0 # shape (batch_size, seq_len)
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    

    #extend stage1 prediction to word unit length
    l = 0
    tr_acc = 0
    for i,j in enumerate(stage1_span_length):
        label_e = labels.tolist()[l:(l+j)]
        prediction_e = predictions.tolist()[l:(l+j)]
        l+=j
        assert j == len(sentence_length_batch[i])
        if label_e == prediction_e:
            tr_acc+=1
    tr_cor = sum(1 for x,y in zip(labels.cpu().numpy(), predictions.cpu().numpy()) if x == y) 
    tr_label_l = len(labels.cpu().numpy())
    
    
    # Run words through encoder
    #encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
    #                                          token_type_ids=token_type_ids_var, sentence_ids=stage1_sentence_ids_var,
    #                                          stage1_ids=extended_predictions_var)
    encoder_outputs = encoder_outputs.transpose(0,1)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]  # root embedding B x 1

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.stage2.config.hidden_size)

    num_start = output_lang.num_start - len(var_nums)
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
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
    
    #print(tr_loss)
    return loss_eq.item(), tr_loss.item(), tr_cor, tr_acc, tr_label_l  # , loss_0.item(), loss_1.item()
    
    
def evaluate_lm2tree_v4(input_batch, input_length, attention_mask_batch, token_type_ids_batch, stage1_span_ids_batch, 
                       stage1_span_length, stage1_sentence_ids_batch, attention_mask_sentence_batch, 
                       sentence_length_batch, quantity_indicator_batch, generate_nums, 
                       encoder, predict, generate, merge, output_lang, num_pos,
                       beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH, labels_num=5):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    stage1_sentence_ids_var = torch.LongTensor(stage1_sentence_ids_batch).unsqueeze(0)
    quantity_ids_var = torch.LongTensor(quantity_indicator_batch).unsqueeze(0)
    stage1_span_ids_var = torch.LongTensor(stage1_span_ids_batch).unsqueeze(0)
    attention_mask_sentence_var = torch.LongTensor(attention_mask_sentence_batch).unsqueeze(0)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)

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
    
    #Stage1 span classification
    #sep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var)
    #nosep version
    #stage1_outputs = stage1_encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, stage1_span_ids_var, [sentence_length_batch])
    
    tr_logits, tr_loss, encoder_outputs, problem_output = encoder(input_var, attention_mask_var, attention_mask_sentence_var, stage1_sentence_ids_var, quantity_ids_var, stage1_span_ids_var, [sentence_length_batch], [stage1_span_length])
    

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
    
    
    
    
def train_graph2tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                     encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
                     merge_optimizer, output_lang, num_pos, batch_graph, var_nums=[], use_clip=False, clip=0.0,
                     grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)  + len(var_nums)
    for i in num_size_batch:
        d = i + len(generate_nums) + len(var_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :] # 因为梯度累计


    batch_graph = torch.LongTensor(batch_graph)
    batch_graph = batch_graph[:,:, :max_len, :max_len]  # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(input_var, input_length, batch_graph)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start - len(var_nums)
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
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

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_graph2tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos, batch_graph, beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length], batch_graph)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
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

        return beams[0].out
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
        return all_node_outputs[0]


def train_lmgraph2tree(input_batch, input_length, attention_mask_batch, token_type_ids_batch, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                  encoder, predict, generate, merge, encoder_bert_optimizer, encoder_graph_optimizer, predict_optimizer, generate_optimizer,
                  merge_optimizer, output_lang, num_pos, batch_graph, var_nums=[], use_clip=False, clip=0.0,
                       grad_acc=False, zero_grad=True, grad_acc_steps=1, english=False):
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
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)
    input_var = torch.LongTensor(input_batch)
    attention_mask_var = torch.LongTensor(attention_mask_batch)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch)

    input_var = input_var[:, :max_len]  # 因为梯度累计
    attention_mask_var = attention_mask_var[:, :max_len]  # 因为梯度累计
    token_type_ids_var = token_type_ids_var[:, :max_len]  # 因为梯度累计

    target = torch.LongTensor(target_batch).transpose(0, 1)
    max_target_len = max(target_length)
    target = target[:max_target_len, :]  # 因为梯度累计

    batch_graph = torch.LongTensor(batch_graph)
    batch_graph = batch_graph[:,:, :max_len, :max_len]  # 因为梯度累计

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        attention_mask_var = attention_mask_var.cuda()
        token_type_ids_var = token_type_ids_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()

    # Zero gradients of both optimizers
    if zero_grad:
        encoder_bert_optimizer.zero_grad()
        encoder_graph_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, batch_graph=batch_graph)
    encoder_outputs = encoder_outputs.transpose(0, 1)
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

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy.deepcopy(nums_stack_batch), num_start, unk)
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

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length) / grad_acc_steps
    # loss = loss_0 + loss_1
    loss.backward()

    # clip the grad
    # if clip > 0:
    if use_clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(merge.parameters(), clip)

    # Update parameters with optimizers
    if not grad_acc:
        encoder_bert_optimizer.step()
        encoder_graph_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_lmgraph2tree(input_batch, input_length, attention_mask_batch, token_type_ids_batch, generate_nums, encoder, predict, generate, merge, output_lang, num_pos, batch_graph,
                     beam_size=5, beam_search=True, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)
    attention_mask_var = torch.LongTensor(attention_mask_batch).unsqueeze(0)
    token_type_ids_var = torch.LongTensor(token_type_ids_batch).unsqueeze(0)
    batch_graph = torch.LongTensor(batch_graph)
    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)

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
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()
    # Run words through encoder

    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    encoder_outputs, problem_output = encoder(input_batch=input_var, attention_mask=attention_mask_var,
                                              token_type_ids=token_type_ids_var, batch_graph=batch_graph)
    encoder_outputs = encoder_outputs.transpose(0,1)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
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

        return beams[0].out
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
        return all_node_outputs[0]


# def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
#                        generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
#                        generate_optimizer, output_lang, num_pos, var_nums=[], use_clip=False, clip=0.0, english=False):
#     # sequence mask for attention
#     seq_mask = []
#     max_len = max(input_length)
#     for i in input_length:
#         seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
#     seq_mask = torch.ByteTensor(seq_mask)
#
#     num_mask = []
#     max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums)
#     for i in num_size_batch:
#         d = i + len(generate_nums) + len(var_nums)
#         num_mask.append([0] * d + [1] * (max_num_size - d))
#     num_mask = torch.ByteTensor(num_mask)
#
#     unk = output_lang.word2index["[UNK]"]
#
#     # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
#     input_var = torch.LongTensor(input_batch).transpose(0, 1)
#
#     target = torch.LongTensor(target_batch).transpose(0, 1)
#
#     padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
#     batch_size = len(input_length)
#
#     encoder.train()
#     predict.train()
#     generate.train()
#
#     if USE_CUDA:
#         input_var = input_var.cuda()
#         seq_mask = seq_mask.cuda()
#         padding_hidden = padding_hidden.cuda()
#         num_mask = num_mask.cuda()
#
#     # Zero gradients of both optimizers
#     encoder_optimizer.zero_grad()
#     predict_optimizer.zero_grad()
#     generate_optimizer.zero_grad()
#     # Run words through encoder
#
#     encoder_outputs, problem_output = encoder(input_var, input_length)
#     # Prepare input and output variables
#     node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
#
#     max_target_length = max(target_length)
#
#     all_node_outputs = []
#     # all_leafs = []
#
#     copy_num_len = [len(_) for _ in num_pos]
#     num_size = max(copy_num_len)
#     all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
#                                                               encoder.hidden_size)
#
#     num_start = output_lang.num_start  - len(var_nums)
#     left_childs = [None for _ in range(batch_size)]
#     for t in range(max_target_length):
#         num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
#             node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
#
#         # all_leafs.append(p_leaf)
#         outputs = torch.cat((op, num_score), 1)
#         all_node_outputs.append(outputs)
#
#         target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
#         target[t] = target_t
#         if USE_CUDA:
#             generate_input = generate_input.cuda()
#         left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
#         for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
#                                             node_stacks, target[t].tolist()):
#             if len(node_stack) != 0:
#                 node = node_stack.pop()
#             else:
#                 continue
#
#             if i < num_start:
#                 node_stack.append(TreeNode(r))
#                 node_stack.append(TreeNode(l, left_flag=True))
#
#     # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
#     all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
#
#     target = target.transpose(0, 1).contiguous()
#     if USE_CUDA:
#         # all_leafs = all_leafs.cuda()
#         all_node_outputs = all_node_outputs.cuda()
#         target = target.cuda()
#
#     # op_target = target < num_start
#     # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
#     loss = masked_cross_entropy(all_node_outputs, target, target_length)
#     # loss = loss_0 + loss_1
#     loss.backward()
#     # clip the grad
#     # if clip > 0:
#     if use_clip:
#         torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
#         torch.nn.utils.clip_grad_norm_(predict.parameters(), clip)
#         torch.nn.utils.clip_grad_norm_(generate.parameters(), clip)
#
#     # Update parameters with optimizers
#     encoder_optimizer.step()
#     predict_optimizer.step()
#     generate_optimizer.step()
#     return loss.item()  # , loss_0.item(), loss_1.item()
#
#
# def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
#                           beam_size=5, var_nums=[], english=False, max_length=MAX_OUTPUT_LENGTH):
#
#     seq_mask = torch.ByteTensor(1, input_length).fill_(0)
#     # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
#     input_var = torch.LongTensor(input_batch).unsqueeze(1)
#
#     num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums) + len(var_nums)).fill_(0)
#
#     # Set to not-training mode to disable dropout
#     encoder.eval()
#     predict.eval()
#     generate.eval()
#
#     padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
#
#     batch_size = 1
#
#     if USE_CUDA:
#         input_var = input_var.cuda()
#         seq_mask = seq_mask.cuda()
#         padding_hidden = padding_hidden.cuda()
#         num_mask = num_mask.cuda()
#     # Run words through encoder
#
#     encoder_outputs, problem_output = encoder(input_var, [input_length])
#
#     # Prepare input and output variables
#     node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
#
#     num_size = len(num_pos)
#     all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
#                                                               encoder.hidden_size)
#     num_start = output_lang.num_start - len(var_nums)
#     # B x P x N
#     embeddings_stacks = [[] for _ in range(batch_size)]
#     left_childs = [None for _ in range(batch_size)]
#
#     beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
#
#     for t in range(max_length):
#         current_beams = []
#         while len(beams) > 0:
#             b = beams.pop()
#             if len(b.node_stack[0]) == 0:
#                 current_beams.append(b)
#                 continue
#             # left_childs = torch.stack(b.left_childs)
#
#             num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
#                 b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
#                 seq_mask, num_mask)
#
#             # leaf = p_leaf[:, 0].unsqueeze(1)
#             # repeat_dims = [1] * leaf.dim()
#             # repeat_dims[1] = op.size(1)
#             # leaf = leaf.repeat(*repeat_dims)
#             #
#             # non_leaf = p_leaf[:, 1].unsqueeze(1)
#             # repeat_dims = [1] * non_leaf.dim()
#             # repeat_dims[1] = num_score.size(1)
#             # non_leaf = non_leaf.repeat(*repeat_dims)
#             #
#             # p_leaf = torch.cat((leaf, non_leaf), dim=1)
#             out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
#
#             # out_score = p_leaf * out_score
#
#             topv, topi = out_score.topk(beam_size)
#
#             # is_leaf = int(topi[0])
#             # if is_leaf:
#             #     topv, topi = op.topk(1)
#             #     out_token = int(topi[0])
#             # else:
#             #     topv, topi = num_score.topk(1)
#             #     out_token = int(topi[0]) + num_start
#
#             for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
#                 current_node_stack = copy_list(b.node_stack)
#                 current_out = copy.deepcopy(b.out)
#
#                 out_token = int(ti)
#                 current_out.append(out_token)
#
#                 node = current_node_stack[0].pop()
#
#                 if out_token < num_start:
#                     generate_input = torch.LongTensor([out_token])
#                     if USE_CUDA:
#                         generate_input = generate_input.cuda()
#                     left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
#
#                     current_node_stack[0].append(TreeNode(right_child))
#                     current_node_stack[0].append(TreeNode(left_child, left_flag=True))
#
#                 current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
#                                               current_out))
#         beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
#         beams = beams[:beam_size]
#         flag = True
#         for b in beams:
#             if len(b.node_stack[0]) != 0:
#                 flag = False
#         if flag:
#             break
#
#     return beams[0].out
