import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel , BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from packaging import version
from src.neural import MultiHeadedAttention, PositionwiseFeedForward
from src.rnn import LayerNormLSTM
from src.masked_cross_entropy import *

import math

class BaseRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers,
                 rnn_dropout=0.5, rnn_cell_name="gru"):
        super(BaseRNN, self).__init__()
        # embedding
        self.embedding_size = embedding_size

        # rnn
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dropout_rate = rnn_dropout
        # self.rnn_dropout = nn.Dropout(self.rnn_dropout_rate)
        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU
        elif rnn_cell_name.lower() == "rnn":
            self.rnn_cell = nn.RNN
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_name))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
        
        
class GeneralEncoderRNN(BaseRNN):
    def __init__(self, embedding_size, hidden_size, n_layers,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=False,
                 bias=True, batch_first=True, rnn_cell_name="gru", max_seq_len=512):
        super(GeneralEncoderRNN, self).__init__(embedding_size=embedding_size,
                                                hidden_size=hidden_size, n_layers=n_layers,
                                                rnn_dropout=rnn_dropout,
                                                rnn_cell_name=rnn_cell_name)
        self.max_seq_len = max_seq_len
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias


        # rnn
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers, dropout=self.rnn_dropout_rate,
                                 bidirectional=self.bidirectional, batch_first=self.batch_first, bias=self.bias)

    def forward(self, inputs_embeds, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        if self.variable_lengths:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(input_embeds, input_lengths, batch_first=self.batch_first)
        outputs, hidden = self.rnn(embedded, hidden)
        if self.variable_lengths:
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)  # unpack (back to padded)
        # fusion strategy
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # batch_first=False: S x B x H
        # batch_first=True: B x S x H
        return outputs
    
    
    
    
    
class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 5, bias=True)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask, labels):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        x = self.wo(x)
        #x = F.gumbel_softmax(x, dim=-1)
        #loss_fct = nn.NLLLoss()
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        #loss = loss_fct(x.log().view(-1, x.size(-1)), labels.view(-1))
        loss = loss_fct(x.view(-1, x.size(-1)), labels.view(-1))
        #sent_scores = F.log_softmax(self.wo(x), dim=-1)
        #sent_scores = self.sigmoid(self.wo(x))
        #sent_scores = sent_scores.squeeze(-1) * mask.float()
        return x, loss
        #return sent_scores


class RNNEncoder(nn.Module):

    def __init__(self, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        """See :func:`EncoderBase.forward()`"""
        x = torch.transpose(x, 1, 0)
        memory_bank, _ = self.rnn(x)
        memory_bank = self.dropout(memory_bank) + x
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.sigmoid(self.wo(memory_bank))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores