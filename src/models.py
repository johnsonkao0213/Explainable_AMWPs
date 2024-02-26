# -*- encoding: utf-8 -*-
# @Author: Jinghui Qin
# @Time: 2021/10/9
# @File: models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from transformers import AutoModel, AutoConfig, BertModel, BertConfig, BertForTokenClassification
from transformers.models.bert.modeling_bert import BertPreTrainedModel , BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from packaging import version
from itertools import groupby
from src.model_stage1 import TransformerInterEncoder, Classifier, RNNEncoder

class CustomBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, sentence_indicator=None, stage1_indicator=None, quantity_indicator=None, equation_index_indicator=None, var_cnt_indicator=None, num_labels=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        #max_number_of_sentence_tags
        if sentence_indicator:
            self.sentence_tag_embeddings = nn.Embedding(9, config.hidden_size)
        if stage1_indicator:
            self.stage1_tag_embeddings = nn.Embedding(num_labels, config.hidden_size)
            #self.weight = torch.randn((5, 768), requires_grad=True)
            #self.stage1_tag_embeddings = torch.randn(requires_grad=True, device=self.device)
        if quantity_indicator:
            self.quantity_tag_embeddings = nn.Embedding(2, config.hidden_size)
        if var_cnt_indicator:
            self.var_cnt_tag_embeddings = nn.Embedding(2, config.hidden_size)
        if equation_index_indicator:
            self.equation_index_tag_embeddings = nn.Embedding(3, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )
  
    def forward(
        self, input_ids=None, sentence_ids=None, token_type_ids=None, position_ids=None, stage1_ids=None, quantity_ids=None, equation_index_ids=None, var_cnt_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        
        #
        #print(token_type_embeddings.shape)
        #print("token_type_embeddings: {}".format(token_type_embeddings.shape))
        embeddings = inputs_embeds + token_type_embeddings
        if sentence_ids is not None:
            sentence_tag_embeddings = self.sentence_tag_embeddings(sentence_ids)
            embeddings += sentence_tag_embeddings
        if stage1_ids is not None:
            stage1_tag_embeddings = self.stage1_tag_embeddings(stage1_ids)
            #print(stage1_ids.shape)
            #weight = torch.rand((5, 768), requires_grad=True, device = stage1_ids.device)
            #print(self.stage1_tag_embeddings.weight.shape)
            #print(self.stage1_tag_embeddings.weight.clone().requires_grad)
            #stage1_tag_embeddings = torch.matmul(stage1_ids, self.stage1_tag_embeddings.weight.clone())
            #stage1_tag_embeddings = torch.matmul(stage1_ids, self.weight.cuda())
            #print(stage1_tag_embeddings.shape)
            #stage1_tag_embeddings = torch.stack([torch.mm(batch.float(), weight) for batch in stage1_ids], dim=0)
            embeddings += stage1_tag_embeddings
        if quantity_ids is not None:
            quantity_tag_embeddings = self.quantity_tag_embeddings(quantity_ids)
            embeddings += quantity_tag_embeddings
        if equation_index_ids is not None:
            equation_index_tag_embeddings = self.equation_index_tag_embeddings(equation_index_ids)
            embeddings += equation_index_tag_embeddings
        if var_cnt_ids is not None:
            var_cnt_tag_embeddings = self.var_cnt_tag_embeddings(var_cnt_ids)
            embeddings += var_cnt_tag_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
    
    
#stage1    
class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomBertEmbeddings(config, sentence_indicator=False, stage1_indicator=True, quantity_indicator=False, num_labels=2)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentence_ids=None,
        quantity_ids=None,
        stage1_ids=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            sentence_ids=sentence_ids,
            quantity_ids=quantity_ids,
            stage1_ids=stage1_ids
        )

        # you can modify the embedding outputs here
        #print(embedding_output.shape)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

#stage2
class CustomBertModel_v2(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomBertEmbeddings(config, sentence_indicator=False, stage1_indicator=True, quantity_indicator=False, num_labels=5)
        #self.device = self.embeddings.device
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentence_ids = None,
        stage1_ids=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            sentence_ids=sentence_ids,
            stage1_ids=stage1_ids
        )

        # you can modify the embedding outputs here
        #print(embedding_output.shape)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    

    
    
class CustomBertModel_v3(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        #self.embeddings = CustomBertEmbeddings(config, sentence_indicator=True, stage1_indicator=True, quantity_indicator=False, num_labels=5)
        self.stage1_tag_embeddings = nn.Embedding(5, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentence_ids = None,
        stage1_ids=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


#         if token_type_ids is None:
#             if hasattr(self.embeddings, "token_type_ids"):
#                 buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    
#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds,
#             past_key_values_length=past_key_values_length,
#             sentence_ids=sentence_ids,
#             stage1_ids=stage1_ids
#         )

        # you can modify the embedding outputs here
        #print(embedding_output.shape)
        stage1_tag_embeddings = self.stage1_tag_embeddings(stage1_ids)
        #stage1_tag_embeddings = torch.matmul(stage1_ids, self.stage1_tag_embeddings.weight.clone())
        inputs_embeds = inputs_embeds + stage1_tag_embeddings
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(inputs_embeds)
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    

class CustomBertModel_v4(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomBertEmbeddings(config, sentence_indicator=False, stage1_indicator=True, quantity_indicator=True, equation_index_indicator=True, var_cnt_indicator=False, num_labels=2)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentence_ids=None,
        quantity_ids=None,
        stage1_ids=None,
        equation_index_ids=None, 
        var_cnt_ids=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            sentence_ids=sentence_ids,
            quantity_ids=quantity_ids,
            stage1_ids=stage1_ids,
            equation_index_ids=equation_index_ids, 
            var_cnt_ids=var_cnt_ids
        )

        # you can modify the embedding outputs here
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
    
    
class Stage1_Encoder(nn.Module):
    def __init__(self, num_labels, tokenizer, use_sentence_index, aggregate_mode, rnn_type):
        super(Stage1_Encoder, self).__init__()
        self.num_labels = num_labels
        self.use_sentence_index = use_sentence_index
        self.aggregate_mode = aggregate_mode
        self.rnn_type = rnn_type
        if self.use_sentence_index:
            self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        #self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        self.bert.resize_token_embeddings(len(tokenizer))
        if rnn_type == 'transformer':
            self.model2 = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                                     num_labels=self.num_labels)
            self.model2.resize_token_embeddings(len(tokenizer))
        elif rnn_type == 'rnn':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="rnn", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'lstm':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="lstm", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'gru':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="gru", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        else:
            raise ValueError("Model {} not found".format(rnn_type))

    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    

    def forward(
            self,
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            labels=None,
        ):
        embeds=[]
        seq_len=[]
        batch_size = input_ids.size(0)
        device = input_ids.device

        for i in range(batch_size):
            #print(input_ids[i])
            #print(attention_mask1[i])
            #print(sentence_ids[i])
            this_attention_mask = attention_mask1[i].unsqueeze(0)
            if self.use_sentence_index:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask,
                    sentence_ids=sentence_ids[i].unsqueeze(0)
                )
                #print(outputs[0])
            else:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask
                )
            sentence_m = [list(group) for k, group in groupby(input_ids[i].tolist(), lambda x: x == 102 or
                                                             x == 0) if not k]
            c = 0
            #masked labels
            mask_labels = labels[i] != -100 # shape (batch_size, seq_len)
            active_labels = torch.masked_select(labels[i], mask_labels)
            seq_len.append(active_labels.size(0))
            tokenized_sentence=[]
            #[TODO]: change into flexible variable
            #print(labels[i].size())
            max_len = labels[i].size(0) #max_problem_length (sentence unit)
            #print(max_len)
            for i, j in zip(sentence_m, active_labels.tolist()):
                l = len(i) #process BPE
                if self.aggregate_mode=='mean':
                    sentence_embeddings = self.mean_pooling(outputs[0][:, c:(c+l), :], 
                                                       this_attention_mask[:, c:(c+l)])
                elif self.aggregate_mode=='cls':
                    sentence_embeddings = outputs[0][:, c, :]
                else: #sep
                    #print(c+l)
                    sentence_embeddings = outputs[0][:, c+l, :]
                c += (l+1)
                tokenized_sentence.extend(sentence_embeddings)
            if len(tokenized_sentence) < max_len:
                padding = -100.0 * torch.ones(768, device=device)
                tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
            tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
            #print(tokenized_sentence.shape)
            embeds.append(tokenized_sentence)
        
        next_inputs_embeds = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        #print(next_inputs_embeds.shape)
        if self.rnn_type == 'transformer':
            outputs = self.model2(inputs_embeds=next_inputs_embeds, attention_mask=attention_mask2)
            pred = F.gumbel_softmax(outputs.logits, dim=-1)
            loss = F.nll_loss(pred, labels)
        else:
            rnn_outputs = self.model2(inputs_embeds=next_inputs_embeds, input_lengths=seq_len)
            rnn_outputs = self.em_dropout(rnn_outputs)
            outputs = self.classifier(rnn_outputs)
                     
        return outputs

    
    
#gumbel version + quantity indicator
class Stage1_Encoder_nosep_v2(nn.Module):
    def __init__(self, num_labels, tokenizer, use_sentence_index, aggregate_mode, rnn_type):
        super(Stage1_Encoder_nosep_v2, self).__init__()
        self.num_labels = num_labels
        self.use_sentence_index = use_sentence_index
        self.aggregate_mode = aggregate_mode
        self.rnn_type = rnn_type
        if self.use_sentence_index:
            self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        #self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        self.bert.resize_token_embeddings(len(tokenizer))
        if rnn_type == 'transformer':
            self.model2 = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                                     num_labels=self.num_labels)
            self.model2.resize_token_embeddings(len(tokenizer))
        elif rnn_type == 'rnn':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="rnn", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'lstm':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="lstm", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'gru':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="gru", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        else:
            raise ValueError("Model {} not found".format(rnn_type))

    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    

    def forward(
            self,
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
        ):
        embeds=[]
        seq_len=[]
        batch_size = input_ids.size(0)
        device = input_ids.device
        #print(sentence_length)
        for i in range(batch_size):
            #print(input_ids[i])
            #print(attention_mask1[i])
            #print(sentence_ids[i])
            this_attention_mask = attention_mask1[i].unsqueeze(0)
            if self.use_sentence_index:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask,
                    sentence_ids=sentence_ids[i].unsqueeze(0)
                )
                #quantity_ids=quantity_ids[i].unsqueeze(0)
                #print(outputs[0])
            else:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask
                )
            #sentence_m = [list(group) for k, group in groupby(input_ids[i].tolist(), lambda x: x == 102 or
            #                                                 x == 0) if not k]
            
            c = 0
            #masked labels
            mask_labels = labels[i] != -100 # shape (batch_size, seq_len)
            active_labels = torch.masked_select(labels[i], mask_labels)
            seq_len.append(active_labels.size(0))
            tokenized_sentence=[]
            #[TODO]: change into flexible variable
            #print(labels[i].size())
            max_len = labels[i].size(0) #max_problem_length (sentence unit)
            for length, j in zip(sentence_length[i], active_labels.tolist()):
                #l = len(i) #process BPE
                #print(length)
                if self.aggregate_mode=='mean':
                    sentence_embeddings = self.mean_pooling(outputs[0][:, c:(c+length), :], 
                                                       this_attention_mask[:, c:(c+length)])
                elif self.aggregate_mode=='cls':
                    sentence_embeddings = outputs[0][:, c, :]
                else: #sep
                    sentence_embeddings = outputs[0][:, c+length, :]
                c += length #nosep
                tokenized_sentence.extend(sentence_embeddings)
            if len(tokenized_sentence) < max_len:
                padding = -100.0 * torch.ones(768, device=device)
                tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
            tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
            #print(tokenized_sentence.shape)
            embeds.append(tokenized_sentence)
        
        next_inputs_embeds = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        #print(next_inputs_embeds.shape)
        if self.rnn_type == 'transformer':
            outputs = self.model2(inputs_embeds=next_inputs_embeds, attention_mask=attention_mask2)
            pred = F.gumbel_softmax(outputs.logits, dim=-1)
            #pred = F.log_softmax(pred, dim=-1)
            loss_fct = nn.NLLLoss()
            #loss = F.nll_loss(pred, labels.view(-1))
            loss = loss_fct(pred.log().view(-1, self.num_labels), labels.view(-1))
            #loss = loss_fct(pred, labels.view(-1))
            #outputs = self.model2(inputs_embeds=next_inputs_embeds, attention_mask=attention_mask2, labels=labels)
        else:
            rnn_outputs = self.model2(inputs_embeds=next_inputs_embeds, input_lengths=seq_len)
            rnn_outputs = self.em_dropout(rnn_outputs)
            outputs = self.classifier(rnn_outputs)
                     
        #return outputs.logits, outputs.loss
        #return outputs.logits, loss
        return outputs.logits, loss, pred
        #return pred, loss
    
    
    
    
#gumbel version + quantity indicator + End2End
class Stage1_Encoder_nosep_v3(nn.Module):
    def __init__(self, num_labels, tokenizer, use_sentence_index, aggregate_mode, rnn_type):
        super(Stage1_Encoder_nosep_v3, self).__init__()
        self.num_labels = num_labels
        self.use_sentence_index = use_sentence_index
        self.aggregate_mode = aggregate_mode
        self.rnn_type = rnn_type
        if self.use_sentence_index:
            self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        #self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        #self.bert.resize_token_embeddings(len(tokenizer))
        if rnn_type == 'transformer':
            self.model2 = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                                     num_labels=self.num_labels)
            #self.model2.resize_token_embeddings(len(tokenizer))
        elif rnn_type == 'rnn':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="rnn", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'lstm':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="lstm", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'gru':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="gru", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        else:
            raise ValueError("Model {} not found".format(rnn_type))
    
        self.stage2 = CustomBertModel_v3.from_pretrained("bert-base-uncased")
        
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    

    def forward(
            self,
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
            stage1_span_length=None
        ):
        embeds=[]
        embedding_outputs=[]
        #seq_len=[]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device
        for i in range(batch_size):
            this_attention_mask = attention_mask1[i].unsqueeze(0)
            if self.use_sentence_index:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask,
                    sentence_ids=sentence_ids[i].unsqueeze(0),
                    quantity_ids=quantity_ids[i].unsqueeze(0),
                    output_hidden_states=True
                )
            else:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask
                )
            
            c = 0
            #masked labels
            mask_labels = labels[i] != -100 # shape (batch_size, seq_len)
            active_labels = torch.masked_select(labels[i], mask_labels)
            #seq_len.append(active_labels.size(0))
            tokenized_sentence=[]
            
            #[TODO]: change into flexible variable
            #print(labels[i].size())
            max_len = labels[i].size(0) #max_problem_length (sentence unit)
            for length, j in zip(sentence_length[i], active_labels.tolist()):
                if self.aggregate_mode=='mean':
                    sentence_embeddings = self.mean_pooling(outputs[0][:, c:(c+length), :], 
                                                       this_attention_mask[:, c:(c+length)])
                elif self.aggregate_mode=='cls':
                    sentence_embeddings = outputs[0][:, c, :]
                else: #sep
                    sentence_embeddings = outputs[0][:, c+length, :]
                c += length #nosep
                tokenized_sentence.extend(sentence_embeddings)
            if len(tokenized_sentence) < max_len:
                padding = -100.0 * torch.ones(768, device=device)
                tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
            tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
            embedding_outputs.append(outputs[2][0].squeeze(0))
            #print(tokenized_sentence.shape)
            embeds.append(tokenized_sentence)
        
        next_inputs_embeds = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        embedding_outputs = torch.stack(embedding_outputs)
        
        if self.rnn_type == 'transformer':
            outputs = self.model2(inputs_embeds=next_inputs_embeds, attention_mask=attention_mask2)
            pred = F.gumbel_softmax(outputs.logits, dim=-1)
            #pred = F.log_softmax(pred, dim=-1)
            loss_fct = nn.NLLLoss()
            #loss = F.nll_loss(pred, labels.view(-1))
            loss = loss_fct(pred.log().view(-1, self.num_labels), labels.view(-1))
            #loss = loss_fct(pred, labels.view(-1))
            #outputs = self.model2(inputs_embeds=next_inputs_embeds, attention_mask=attention_mask2, labels=labels)
        else:
            rnn_outputs = self.model2(inputs_embeds=next_inputs_embeds, input_lengths=seq_len)
            rnn_outputs = self.em_dropout(rnn_outputs)
            outputs = self.classifier(rnn_outputs)
        
        stage1_indicator=[]
        for i,j in enumerate(stage1_span_length):
            extended_pred = torch.repeat_interleave(pred[i, :j, :],
                                                    torch.tensor(sentence_length[i], device=device), dim=0)
            padder = torch.zeros((seq_len-extended_pred.size(0), extended_pred.size(1)), device=device)
            padded_extended_prediction = torch.cat([extended_pred, padder], dim=0)
            stage1_indicator.append(padded_extended_prediction)
        
        stage1_indicator_var = torch.stack(stage1_indicator)
        stage1_indicator_var = stage1_indicator_var[:, :seq_len]
        stage1_indicator_var = stage1_indicator_var.cuda()
        lm_output = self.stage2(inputs_embeds=embedding_outputs, attention_mask=attention_mask1, stage1_ids=stage1_indicator_var)
        encoder_outputs = lm_output[0]
        problem_output = lm_output[1]
        return outputs.logits, loss, encoder_outputs, problem_output
        
    
    
class Stage1_Encoder_nosep(nn.Module):
    def __init__(self, num_labels, tokenizer, use_sentence_index, aggregate_mode, rnn_type):
        super(Stage1_Encoder_nosep, self).__init__()
        self.num_labels = num_labels
        self.use_sentence_index = use_sentence_index
        self.aggregate_mode = aggregate_mode
        self.rnn_type = rnn_type
        if self.use_sentence_index:
            self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        #self.bert = CustomBertModel.from_pretrained("bert-base-uncased")
        #self.bert.resize_token_embeddings(len(tokenizer))
        if rnn_type == 'transformer':
            self.model2 = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                                     num_labels=self.num_labels)
            #self.model2.resize_token_embeddings(len(tokenizer))
        elif rnn_type == 'rnn':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="rnn", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'lstm':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="lstm", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        elif rnn_type == 'gru':
            self.model2 = GeneralEncoderRNN(768, 768, 4,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=True,
                 bias=True, batch_first=True, rnn_cell_name="gru", max_seq_len=MAX_LEN)
            self.em_dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, self.num_labels)
        else:
            raise ValueError("Model {} not found".format(rnn_type))

    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    

    def forward(
            self,
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
        ):
        embeds=[]
        seq_len=[]
        batch_size = input_ids.size(0)
        device = input_ids.device
        for i in range(batch_size):
            this_attention_mask = attention_mask1[i].unsqueeze(0)
            if self.use_sentence_index:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask,
                    sentence_ids=sentence_ids[i].unsqueeze(0),
                    quantity_ids=quantity_ids[i].unsqueeze(0)
                )
            else:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=this_attention_mask
                )
            #sentence_m = [list(group) for k, group in groupby(input_ids[i].tolist(), lambda x: x == 102 or
            #                                                 x == 0) if not k]
            
            c = 0
            #masked labels
            mask_labels = labels[i] != -100 # shape (batch_size, seq_len)
            active_labels = torch.masked_select(labels[i], mask_labels)
            seq_len.append(active_labels.size(0))
            tokenized_sentence=[]
            #[TODO]: change into flexible variable
            #print(labels[i].size())
            max_len = labels[i].size(0) #max_problem_length (sentence unit)
            for length, j in zip(sentence_length[i], active_labels.tolist()):
                #l = len(i) #process BPE
                #print(length)
                if self.aggregate_mode=='mean':
                    sentence_embeddings = self.mean_pooling(outputs[0][:, c:(c+length), :], 
                                                       this_attention_mask[:, c:(c+length)])
                elif self.aggregate_mode=='cls':
                    sentence_embeddings = outputs[0][:, c, :]
                else: #sep
                    sentence_embeddings = outputs[0][:, c+length, :]
                c += length #nosep
                tokenized_sentence.extend(sentence_embeddings)
            if len(tokenized_sentence) < max_len:
                padding = -100.0 * torch.ones(768, device=device)
                tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
            tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
            #print(tokenized_sentence.shape)
            embeds.append(tokenized_sentence)
        
        next_inputs_embeds = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        #print(next_inputs_embeds.shape)
        if self.rnn_type == 'transformer':
            outputs = self.model2(inputs_embeds=next_inputs_embeds, attention_mask=attention_mask2, labels=labels)
        else:
            rnn_outputs = self.model2(inputs_embeds=next_inputs_embeds, input_lengths=seq_len)
            rnn_outputs = self.em_dropout(rnn_outputs)
            outputs = self.classifier(rnn_outputs)
                     
        return outputs
    

    
    
class Summarizer_v2(nn.Module):
    def __init__(self, rnn_type, bert_config = None):
        super(Summarizer_v2, self).__init__()
        self.bert = CustomBertModel.from_pretrained('bert-base-uncased')
        if (rnn_type == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(rnn_type=='transformer'):
            self.encoder = TransformerInterEncoder(768, 512, 4,
                                                   0.1, 2)
        elif(rnn_type=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif(rnn_type == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.bert.model = BertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        
        #self.stage2 = CustomBertModel_v3.from_pretrained("bert-base-uncased")
        #self.stage2 = BertModel.from_pretrained("bert-base-uncased")

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)
    
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, 
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
            stage1_span_length=None,
            sep_loc=None):
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        

#         vec = self.bert(input_ids, 
#                         sentence_ids=sentence_ids, 
#                         quantity_ids=quantity_ids, 
#                         attention_mask=attention_mask1,
#                         output_hidden_states=True)
        #vec = self.bert(input_ids, quantity_ids=quantity_ids, attention_mask=attention_mask1)
#         top_vec = vec[0]
        
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), sep_loc]

        embeds=[]
        for i in range(batch_size):
            this_attention_mask = attention_mask1[i].unsqueeze(0)
            outputs = self.bert(
                input_ids[i].unsqueeze(0),
                attention_mask=this_attention_mask
            )
            #sentence_ids=sentence_ids[i].unsqueeze(0)
            #quantity_ids=quantity_ids[i].unsqueeze(0)
            c = 0
            tmp = -1
            tokenized_sentence=[]
            max_len = labels[i].size(0) #max_problem_length (sentence unit)
            for length, j in zip(sentence_length[i], sep_loc[i]):
                assert length == (j-tmp)
                sentence_embeddings = self.mean_pooling(outputs[0][:, c:(j+1), :], 
                                                   this_attention_mask[:, c:(j+1)])
                c += length
                tmp = j
                tokenized_sentence.extend(sentence_embeddings)
            if len(tokenized_sentence) < max_len:
                padding = -100.0 * torch.ones(768, device=device)
                tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
            tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
            embeds.append(tokenized_sentence)
        sents_vec = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        sents_vec = sents_vec * attention_mask2[:, :, None].float()
        sent_scores, loss = self.encoder(sents_vec, attention_mask2, labels)
        
        return sent_scores, loss
    
    
    
    
    
    
class Summarizer(nn.Module):
    def __init__(self, rnn_type, bert_config = None):
        super(Summarizer, self).__init__()
        self.bert = CustomBertModel.from_pretrained('bert-base-uncased')
        if (rnn_type == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(rnn_type=='transformer'):
            self.encoder = TransformerInterEncoder(768, 512, 4,
                                                   0.1, 2)
        elif(rnn_type=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif(rnn_type == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.bert.model = BertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        
        #self.stage2 = CustomBertModel_v3.from_pretrained("bert-base-uncased")
        #self.stage2 = BertModel.from_pretrained("bert-base-uncased")
        #self.var_count_expand = nn.Linear(768, 2048)
        #self.var_count_predict = nn.Linear(2048, 2)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)
    
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
#     def _predict_var_count(self, encoded):
#         # Value should be at least 1.0
#         return F.log_softmax(self.var_count_predict(F.relu(self.var_count_expand(encoded[:, 0]))), dim=-1)
    
    def forward(self, 
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
            stage1_span_length=None,
            sep_loc=None):
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
#         for i in range(batch_size):
#             this_attention_mask = attention_mask1[i].unsqueeze(0)
#             outputs = self.bert(
#                 input_ids[i].unsqueeze(0),
#                 attention_mask=this_attention_mask,
#                 sentence_ids=sentence_ids[i].unsqueeze(0),
#                 quantity_ids=quantity_ids[i].unsqueeze(0)
#             )
            
#             c = 0
#             tmp = -1
#             tokenized_sentence=[]
#             max_len = labels[i].size(0) #max_problem_length (sentence unit)
#             for length, j in zip(sentence_length[i], sep_loc[i]):
#                 assert length = (j-tmp)
#                 sentence_embeddings = self.mean_pooling(outputs[0][:, c:(j+1), :], 
#                                                    this_attention_mask[:, c:(j+1)])
#                 c += length
#                 tmp = j
#                 tokenized_sentence.extend(sentence_embeddings)
#             if len(tokenized_sentence) < max_len:
#                 padding = -100.0 * torch.ones(768, device=device)
#                 tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
#             tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
#             embeds.append(tokenized_sentence)
#         sents_vec = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        
#         vec = self.bert(input_ids, 
#                         sentence_ids=sentence_ids, 
#                         quantity_ids=quantity_ids, 
#                         attention_mask=attention_mask1,
#                         output_hidden_states=True)
        #vec = self.bert(input_ids, attention_mask=attention_mask1)
        vec = self.bert(input_ids, sentence_ids=sentence_ids, attention_mask=attention_mask1)
        top_vec = vec[0]
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), sep_loc]
        sents_vec = sents_vec * attention_mask2[:, :, None].float()
        sent_scores, loss = self.encoder(sents_vec, attention_mask2, labels)
        
        return sent_scores, loss
        #print(stage1_span_length)
        #pred = sent_scores.clone()
#         active_logits = sent_scores.view(-1, sent_scores.size(-1)) # shape (batch_size * seq_len, num_labels)
#         flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
#         active_accuracy = labels.view(-1) != -100.0 # shape (batch_size, seq_len)
#         predictions = torch.masked_select(flattened_predictions, active_accuracy)
#         stage1_indicator=[]
        
        
#         torch.autograd.set_detect_anomaly(True)
#         for i,j in enumerate(stage1_span_length):
#             extended_pred = torch.repeat_interleave(sent_scores[i, :j, :],
#                                                     torch.tensor(sentence_length[i], device=device), dim=0)
#             padder = torch.zeros((seq_len-extended_pred.size(0), extended_pred.size(1)), device=device)
#             padded_extended_prediction = torch.cat([extended_pred, padder], dim=0)
#             stage1_indicator.append(padded_extended_prediction)
        
#         l=0
#         for i,j in enumerate(stage1_span_length):
#             prediction_e = predictions.tolist()[l:(l+j)]
#             l+=j
#             extended_prediction = []
#             #print(prediction_e)
#             for k,v in zip(prediction_e, sentence_length[i]):
#                 extended_prediction.extend([k] * v)
#             extended_prediction.extend([0] * (seq_len-len(extended_prediction)))
#             stage1_indicator.append(extended_prediction)
        
#         #stage1_indicator_var = torch.stack(stage1_indicator)
#         stage1_indicator_var = torch.LongTensor(stage1_indicator)
#         stage1_indicator_var = stage1_indicator_var[:, :seq_len]
#         stage1_indicator_var = stage1_indicator_var.cuda()
#         lm_output = self.stage2(inputs_embeds=vec[2][0], 
#                                 attention_mask=attention_mask1,
#                                 stage1_ids=stage1_indicator_var
#                                 )
#         encoder_outputs = lm_output[0]
#         problem_output = lm_output[1]
#         return sent_scores, loss, encoder_outputs, problem_output
    
    
class Summarizer_v3(nn.Module):
    def __init__(self, rnn_type, bert_config = None):
        super(Summarizer_v3, self).__init__()
        self.bert = CustomBertModel.from_pretrained('bert-base-uncased')
        if (rnn_type == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(rnn_type=='transformer'):
            self.encoder = TransformerInterEncoder(768, 512, 4,
                                                   0.1, 2)
        elif(rnn_type=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif(rnn_type == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.bert.model = BertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        
        #self.stage2 = CustomBertModel_v3.from_pretrained("bert-base-uncased")
        #self.stage2 = BertModel.from_pretrained("bert-base-uncased")
        self.var_count_expand = nn.Linear(768, 2048)
        self.var_count_predict = nn.Linear(2048, 2)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)
    
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _predict_var_count(self, encoded, var_cnt_tensor):
        #print(var_cnt_tensor)
        # Value should be at least 1.0
        logit = self.var_count_predict(F.relu(self.var_count_expand(encoded[:, 0]))) #First token
        loss_fct = nn.CrossEntropyLoss()  
        loss = loss_fct(logit.view(-1, logit.size(-1)), var_cnt_tensor.view(-1))
        return logit, loss
    
    def forward(self, 
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
            stage1_span_length=None,
            sep_loc=None,
            var_cnt_tensor=None):
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
#         for i in range(batch_size):
#             this_attention_mask = attention_mask1[i].unsqueeze(0)
#             outputs = self.bert(
#                 input_ids[i].unsqueeze(0),
#                 attention_mask=this_attention_mask,
#                 sentence_ids=sentence_ids[i].unsqueeze(0),
#                 quantity_ids=quantity_ids[i].unsqueeze(0)
#             )
            
#             c = 0
#             tmp = -1
#             tokenized_sentence=[]
#             max_len = labels[i].size(0) #max_problem_length (sentence unit)
#             for length, j in zip(sentence_length[i], sep_loc[i]):
#                 assert length = (j-tmp)
#                 sentence_embeddings = self.mean_pooling(outputs[0][:, c:(j+1), :], 
#                                                    this_attention_mask[:, c:(j+1)])
#                 c += length
#                 tmp = j
#                 tokenized_sentence.extend(sentence_embeddings)
#             if len(tokenized_sentence) < max_len:
#                 padding = -100.0 * torch.ones(768, device=device)
#                 tokenized_sentence.extend((max_len-len(tokenized_sentence)) * [padding])
#             tokenized_sentence = torch.stack(tokenized_sentence) #[seq_len, 768]
#             embeds.append(tokenized_sentence)
#         sents_vec = pad_sequence(embeds, batch_first=True, padding_value=-100.0)
        
#         vec = self.bert(input_ids, 
#                         sentence_ids=sentence_ids, 
#                         quantity_ids=quantity_ids, 
#                         attention_mask=attention_mask1,
#                         output_hidden_states=True)
        vec = self.bert(input_ids, attention_mask=attention_mask1)
        #vec = self.bert(input_ids, quantity_ids=quantity_ids, attention_mask=attention_mask1)
        top_vec = vec[0]
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), sep_loc]
        sents_vec = sents_vec * attention_mask2[:, :, None].float()
        sent_scores, loss = self.encoder(sents_vec, attention_mask2, labels)
        var_len_tensor, var_len_loss = self._predict_var_count(top_vec, var_cnt_tensor)
        
        return sent_scores, loss, var_len_tensor, var_len_loss
    

class Summarizer_v4(nn.Module):
    def __init__(self, rnn_type, bert_config = None):
        super(Summarizer_v4, self).__init__()
        self.bert = CustomBertModel.from_pretrained('bert-base-uncased')
        if (rnn_type == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(rnn_type=='transformer'):
            self.equation1_encoder = TransformerInterEncoder(768, 512, 4,
                                                   0.1, 2)
            self.equation2_encoder = TransformerInterEncoder(768, 512, 4,
                                                   0.1, 2)
            self.equationQ_encoder = TransformerInterEncoder(768, 512, 4,
                                                   0.1, 2)
        elif(rnn_type=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif(rnn_type == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.bert.model = BertModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        
        #self.stage2 = CustomBertModel_v3.from_pretrained("bert-base-uncased")
        #self.stage2 = BertModel.from_pretrained("bert-base-uncased")
        self.var_count_expand = nn.Linear(768, 2048)
        self.var_count_predict = nn.Linear(2048, 2)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)
    
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _predict_var_count(self, encoded, var_cnt_tensor):
        # Value should be at least 1.0
        logit = self.var_count_predict(F.relu(self.var_count_expand(encoded[:, 0]))) #First token
        loss_fct = nn.CrossEntropyLoss()  
        loss = loss_fct(logit.view(-1, logit.size(-1)), var_cnt_tensor.view(-1))
        return logit, loss
    
    def forward(self, 
            input_ids=None,
            attention_mask1=None,
            attention_mask2=None,
            sentence_ids=None,
            quantity_ids=None,
            labels=None,
            sentence_length=None,
            stage1_span_length=None,
            sep_loc=None,
            var_cnt_tensor=None):
        device = input_ids.device
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        masks_1 = (labels == 1) | (labels == 3)
        masks_2 = (labels == 2) | (labels == 3)
        masks_Q = (labels == 4)
        labels_1 = masks_1.long()
        labels_2 = masks_2.long()
        labels_Q = masks_Q.long()
#         vec = self.bert(input_ids, 
#                         sentence_ids=sentence_ids, 
#                         quantity_ids=quantity_ids, 
#                         attention_mask=attention_mask1,
#                         output_hidden_states=True)
        vec = self.bert(input_ids, attention_mask=attention_mask1)
        #vec = self.bert(input_ids, quantity_ids=quantity_ids, attention_mask=attention_mask1)
        top_vec = vec[0]
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), sep_loc]
        sents_vec = sents_vec * attention_mask2[:, :, None].float()
        sent_scores_1, loss_1 = self.equation1_encoder(sents_vec, attention_mask2, labels_1)
        sent_scores_2, loss_2 = self.equation2_encoder(sents_vec, attention_mask2, labels_2)
        sent_scores_Q, loss_Q = self.equationQ_encoder(sents_vec, attention_mask2, labels_Q)
        var_len_tensor, var_len_loss = self._predict_var_count(top_vec, var_cnt_tensor)
        # compute training accuracy
        sent_scores_1 = sent_scores_1.view(-1, 2) # shape (batch_size * seq_len, num_labels)
        sent_scores_2 = sent_scores_2.view(-1, 2) # shape (batch_size * seq_len, num_labels)
        sent_scores_Q = sent_scores_Q.view(-1, 2) # shape (batch_size * seq_len, num_labels)
        flattened_predictions_1 = torch.argmax(sent_scores_1, axis=1) # shape (batch_size * seq_len,)
        flattened_predictions_2 = torch.argmax(sent_scores_2, axis=1) # shape (batch_size * seq_len,)
        flattened_predictions_Q = torch.argmax(sent_scores_Q, axis=1) # shape (batch_size * seq_len,)
        flattened_targets = labels.view(-1)
        flattened_predictions = torch.zeros_like(flattened_predictions_1)
        for i in range(batch_size*seq_len):
            if flattened_predictions_1[i]==1 and flattened_predictions_2[i]==1:
                flattened_predictions[i] == 3
            else:
                if flattened_predictions_1[i] == 1:
                    flattened_predictions[i] == 1
                elif flattened_predictions_2[i] == 1:
                    flattened_predictions[i] == 2
                else:
                    if flattened_predictions_Q[i] == 1:
                        flattened_predictions[i] == 4
                    else:
                        flattened_predictions[i] == 0
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100.0 # shape (batch_size, seq_len)
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        #compute var cnt accuracy
        var_cnt_pred = var_len_tensor.view(-1, 2) # shape (batch_size * seq_len, num_labels)
        var_cnt_pred = torch.argmax(var_cnt_pred, axis=1) # shape (batch_size * seq_len,)
        
        loss = loss_1+loss_2+loss_Q+var_len_loss
        return predictions, labels, var_cnt_pred, loss
    
    
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, embedding_weight=None):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if embedding_weight is not None:
            if isinstance(embedding_weight, torch.Tensor):
                self.embedding.weight.data.copy_(embedding_weight)
                # self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout,
                          batch_first=False, bidirectional=False)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(1)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class PLMEncoderSeq(nn.Module):
    def __init__(self, model_path):
        super(PLMEncoderSeq, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_batch, attention_mask, token_type_ids):
        lm_output = self.model(input_batch, attention_mask, token_type_ids)
        encoder_outputs = lm_output[0]
        problem_output = lm_output[1]
        return encoder_outputs, problem_output

class PLMEncoderSeq_v2(nn.Module):
    def __init__(self, model_path):
        super(PLMEncoderSeq_v2, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        #self.model = CustomBertModel_v2.from_pretrained(model_path)
        self.model = CustomBertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_batch, attention_mask, token_type_ids, sentence_ids, quantity_ids, stage1_ids):
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_ids=sentence_ids, quantity_ids=quantity_ids, stage1_ids=stage1_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_ids=sentence_ids, stage1_ids=stage1_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids)
        lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids)
        encoder_outputs = lm_output[0]
        problem_output = lm_output[1]
        return encoder_outputs, problem_output

#new version + sentence_ids
class PLMEncoderSeq_v3(nn.Module):
    def __init__(self, model_path):
        super(PLMEncoderSeq_v3, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = CustomBertModel_v2.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_batch, attention_mask, token_type_ids, sentence_ids, stage1_ids):
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_ids=sentence_ids, stage1_ids=stage1_ids)
        lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids)
        encoder_outputs = lm_output[0]
        problem_output = lm_output[1]
        return encoder_outputs, problem_output
    

class PLMEncoderSeq_v4(nn.Module):
    def __init__(self, model_path):
        super(PLMEncoderSeq_v4, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        #self.model = BertModel.from_pretrained(model_path)
        #self.model = CustomBertModel.from_pretrained(model_path)
        self.model = CustomBertModel_v4.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False
        self.linear1 = nn.Linear(9*768, 768)
        #self.linear3 = nn.Linear(9*768, 768)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.drop1 = nn.Dropout(p=0.1)
        #self.linear2 = nn.Linear(3*768, 768)
        #self.embeddings = nn.Embedding(3, 768)
        #self.embeddings2 = nn.Embedding(2, 768)
        #self.eq_embeddings = nn.Embedding(output_size, 768, padding_idx=0)
        #self.em_dropout = nn.Dropout(dropout)

    def forward(self, input_batch, attention_mask, token_type_ids, sentence_ids, quantity_ids, stage1_ids, cls_loc, attention_mask2, equation_index_ids, var_cnt_ids, output_eq_ids): #output_eq_ids
        #var_cnt_tensor = var_cnt_ids.unsqueeze(-1).expand_as(input_batch)
        equation_id_var = equation_index_ids.unsqueeze(-1).expand_as(input_batch)
        
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_ids=sentence_ids, stage1_ids=stage1_ids, equation_index_ids=equation_index_ids, var_cnt_ids=var_cnt_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_index_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_index_ids, var_cnt_ids=var_cnt_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_id_var)
        lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_id_var, quantity_ids=output_eq_ids)
        top_vec = lm_output[0]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_loc]
        #equation_index = self.embeddings(equation_index_ids)
#         if output_eq_ids is None:
#             input_shape = output_eq_ids.size()
#             batch_size, seq_length = input_batch
#             device = output_eq_ids.device
#             output_eq_1st = torch.zeros(((batch_size, seq_length)), device=device)
#         else:
#             output_eq_1st = self.eq_embeddings(output_eq_ids)
#         output_eq_1st = self.em_dropout(output_eq_1st)
        
        #var_cnt = self.embeddings2(var_cnt_ids)
        #sents_vec = sents_vec + equation_index.unsqueeze(1) + var_cnt.unsqueeze(1)
        #sents_vec = sents_vec + equation_index.unsqueeze(1)
        sents_vec2 = sents_vec * attention_mask2[:, :, None].float()
        x = sents_vec2.view(top_vec.size(0), -1)
        
        #x = torch.cat((x, equation_index, var_cnt), dim=1)
        #x = torch.cat((x, equation_index), dim=1)
        #x = torch.cat((equation_index, x), dim=1)
        #x = self.drop1(x)
        #x = self.linear1(x)
        encoder_outputs = lm_output[0]
        #encoder_outputs = stage1_ids.unsqueeze(-1) * lm_output[0]
        problem_output = self.linear1(x)
        
        #problem_output = lm_output[1]
        return encoder_outputs, problem_output    
    
class Eq1_Encoder(nn.Module):
    def __init__(self, hidden_size=768, n_layers=2, dropout=0.5):
        super(Eq1_Encoder, self).__init__()
        #RNN
        self.hidden_size = hidden_size
        self.gru_pade = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*hidden_size, hidden_size)
        
    def forward(self, problem_output, all_node_outputs, target_lengths, pade_hidden=None):
        #Bi-GRU (plus)
        packed = torch.nn.utils.rnn.pack_padded_sequence(all_node_outputs, target_lengths, batch_first=True)
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=True)
        eq1_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        #concat+FF
        fuse = torch.cat((problem_output, eq1_output), 1)
        fuse = self.linear(fuse)
        return fuse

class Eq1_Encoder_v2(nn.Module):
    def __init__(self, input_size, op_nums, hidden_size=768, n_layers=2, dropout=0.5):
        super(Eq1_Encoder_v2, self).__init__()
        #RNN
        self.op_nums = op_nums
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.op_embeddings = nn.Embedding(op_nums, hidden_size)
        self.gru_pade = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*hidden_size, hidden_size)
        
    def forward(self, problem_output, encoder_outputs, prev_eq, num_start, num_pos, target_lengths, pade_hidden=None):
        batch_size = problem_output.size(0)
        prev_eq_input = copy.deepcopy(prev_eq)
#         for b in range(batch_size):
#             for i in range(len(prev_eq[b])):
#                 if prev_eq[b][i] >= num_start:
#                     prev_eq_input[b][i] = 0
        mask = prev_eq >= num_start
        prev_eq_input[mask] = 0
        prev_eq_embed = self.op_embeddings(prev_eq_input)
        encoder_outputs_freeze = encoder_outputs.detach()
        for b in range(batch_size):
            cnt = 0
            for i in range(len(prev_eq[b])):
                if prev_eq[b][i] >= num_start+self.input_size:
                    prev_eq_embed[b][i] = encoder_outputs_freeze[b][num_pos[b][cnt]]
                    cnt+=1
                elif prev_eq[b][i] >= num_start and prev_eq[b][i] < num_start+self.input_size:
                    prev_eq_embed[b][i] = self.embedding_weight[prev_eq[b][i]-num_start]
            assert cnt == len(num_pos[b])
        
        #Bi-GRU (plus)
        packed = torch.nn.utils.rnn.pack_padded_sequence(prev_eq_embed, target_lengths, batch_first=True, enforce_sorted=False)
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=True)
        eq1_output = pade_outputs[:, -1, :self.hidden_size] + pade_outputs[:, 0, self.hidden_size:]
        #concat+FF
        fuse = torch.cat((problem_output, eq1_output), 1)
        fuse = self.linear(fuse)
        return fuse

    
class PLMEncoderSeq_v5(nn.Module):
    def __init__(self, model_path, output_size):
        super(PLMEncoderSeq_v5, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        #self.model = BertModel.from_pretrained(model_path)
        #self.model = CustomBertModel.from_pretrained(model_path)
        self.model = CustomBertModel_v4.from_pretrained(model_path)
        self.linear1 = nn.Linear(9*768, 768)
        #self.linear3 = nn.Linear(9*768, 768)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.drop1 = nn.Dropout(p=0.1)
        #self.linear2 = nn.Linear(3*768, 768)
        #self.embeddings = nn.Embedding(3, 768)
        #self.embeddings2 = nn.Embedding(2, 768)
        self.eq_embeddings = nn.Embedding(output_size, 768, padding_idx=0)
        self.eq1_encoder = Eq1_Encoder()

    def forward(self, input_batch, attention_mask, token_type_ids, sentence_ids, quantity_ids, stage1_ids, cls_loc, attention_mask2, equation_index_ids, var_cnt_ids, output_eq_ids): 
        #var_cnt_tensor = var_cnt_ids.unsqueeze(-1).expand_as(input_batch)
        equation_id_var = equation_index_ids.unsqueeze(-1).expand_as(input_batch)
        
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_ids=sentence_ids, stage1_ids=stage1_ids, equation_index_ids=equation_index_ids, var_cnt_ids=var_cnt_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_index_ids)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_index_ids, var_cnt_ids=var_cnt_ids)
        lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_id_var)
        #lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, stage1_ids=stage1_ids, equation_index_ids=equation_id_var, var_cnt_ids=var_cnt_tensor)
        top_vec = lm_output[0]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_loc]
        #equation_index = self.embeddings(equation_index_ids)
        if output_eq_ids is None:
            input_shape = output_eq_ids.size()
            batch_size, seq_length = input_batch
            device = output_eq_ids.device
            output_eq_1st = torch.zeros(((batch_size, seq_length)), device=device)
        else:
            output_eq_1st = self.eq_embeddings(output_eq_ids)
        
        #var_cnt = self.embeddings2(var_cnt_ids)
        #sents_vec = sents_vec + equation_index.unsqueeze(1) + var_cnt.unsqueeze(1)
        #sents_vec = sents_vec + equation_index.unsqueeze(1)
        sents_vec2 = sents_vec * attention_mask2[:, :, None].float()
        x = sents_vec2.view(top_vec.size(0), -1)
        
        #x = torch.cat((x, equation_index, var_cnt), dim=1)
        #x = torch.cat((x, equation_index), dim=1)
        #x = torch.cat((equation_index, x), dim=1)
        #x = self.drop1(x)
        #x = self.linear1(x)
        encoder_outputs = lm_output[0]
        #encoder_outputs = stage1_ids.unsqueeze(-1) * lm_output[0]
        problem_output = self.linear1(x)
        problem_output2 = self.eq1_encoder(problem_output, output_eq_1st)
        #problem_output = lm_output[1]
        return encoder_outputs, problem_output2
    
    
    
    
class PLMMeanEncoderSeq(nn.Module):
    def __init__(self, model_path):
        super(PLMMeanEncoderSeq, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_batch, attention_mask, token_type_ids):
        lm_output = self.model(input_batch, attention_mask, token_type_ids)
        encoder_outputs = lm_output[0]
        problem_output = self._mean_pooling(lm_output, attention_mask)
        return encoder_outputs, problem_output

class PLMMeanEncoderSeq_v2(nn.Module):
    def __init__(self, model_path):
        super(PLMMeanEncoderSeq_v2, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        #self.model = BertModel.from_pretrained(model_path)
        self.model = CustomBertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_batch, attention_mask, token_type_ids, stage1_span_ids):
        lm_output = self.model(input_ids=input_batch, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_ids=stage1_span_ids)
        encoder_outputs = lm_output[0]
        problem_output = self._mean_pooling(lm_output, attention_mask)
        return encoder_outputs, problem_output


class PLMGraphEncoderSeq(nn.Module):
    def __init__(self, model_path):
        super(PLMGraphEncoderSeq, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

        self.gcn = Graph_Module(self.config.hidden_size, self.config.hidden_size, self.config.hidden_size)

    def forward(self, input_batch, attention_mask, token_type_ids, batch_graph):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        lm_output = self.model(input_batch, attention_mask, token_type_ids)
        encoder_outputs = lm_output[0]  # B x S x H
        problem_output = lm_output[1]  # B x H
        # return encoder_outputs, problem_output

        # problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        _, pade_outputs = self.gcn(encoder_outputs, batch_graph)
        return pade_outputs, problem_output  #problem_output

class PLMGraphMeanEncoderSeq(nn.Module):
    def __init__(self, model_path):
        super(PLMGraphMeanEncoderSeq, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

        self.gcn = Graph_Module(self.config.hidden_size, self.config.hidden_size, self.config.hidden_size)

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_batch, attention_mask, token_type_ids, batch_graph):
        lm_output = self.model(input_batch, attention_mask, token_type_ids)
        encoder_outputs = lm_output[0]
        # problem_output = lm_output[1]  # B x H

        _, pade_outputs = self.gcn(encoder_outputs, batch_graph)

        problem_output = self._mean_pooling(pade_outputs, attention_mask)
        return encoder_outputs, problem_output


class MVPLMEncoderSeq(nn.Module):
    def __init__(self, model_path):
        super(MVPLMEncoderSeq, self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.position_embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.model.embeddings.token_type_embeddings.parameters():
        #     param.requires_grad = False

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_batch, attention_mask, token_type_ids):
        lm_output = self.model(input_batch, attention_mask, token_type_ids)
        encoder_outputs = lm_output[0]
        problem_output1 = lm_output[1]
        problem_output2 = self._mean_pooling(lm_output, attention_mask)
        return encoder_outputs, problem_output1, problem_output2

    
class S2TEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, embedding_weight=None):
        super(S2TEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if embedding_weight is not None:
            if isinstance(embedding_weight, torch.Tensor):
                self.embedding.weight.data.copy_(embedding_weight)
                # self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:] # B x H
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output


class G2TEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, embedding_weight=None):
        super(G2TEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if embedding_weight is not None:
            if isinstance(embedding_weight, torch.Tensor):
                self.embedding.weight.data.copy_(embedding_weight)
                # self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output


class G2SEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, embedding_weight=None):
        super(G2SEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if embedding_weight is not None:
            if isinstance(embedding_weight, torch.Tensor):
                self.embedding.weight.data.copy_(embedding_weight)
                # self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        # problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, pade_hidden #problem_output


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums # 数字列表长度

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)  # left inner symbols generation
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)  # right inner symbols generation
        self.concat_lg = nn.Linear(hidden_size, hidden_size)  # left number generation
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)  # right number generation

        # 用于操作符选择
        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    #def forward(self, external_embedding_weight, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []
        # node_stacks: B
        # padded_hidden: B x 2H
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE
        repeat_dims = [1] * self.embedding_weight.dim()
        #external_embedding_weight = external_embedding_weight.unsqueeze(0)
        #repeat_dims = [1] * external_embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        #embedding_weight = external_embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
    def forward(self, node_embedding, node_label, current_context):
    #def forward(self, op_embeddings, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        #node_label_ = op_embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Graph_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim
        #self.combined_dim = outdim

        #self.edge_layer_1 = nn.Linear(indim, outdim)
        #self.edge_layer_2 = nn.Linear(outdim, outdim)

        #self.dropout = nn.Dropout(p=dropout)
        #self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        #self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)
        self.h = 4
        self.d_k = outdim//self.h

        #layer = GCN(indim, hiddim, self.d_k, dropout)
        self.graph = clones(GCN(indim, hiddim, self.d_k, dropout), 4)

        #self.Graph_0 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_1 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_2 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_3 = GCN(indim, hiddim, outdim//4, dropout)

        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))

        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix

    def normalize(self, A, symmetric=True):
        '''
        ## Inputs:
        - adjacency matrix (K, K) : A
        ## Returns:
        - adjacency matrix (K, K)
        '''
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(A)

    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        '''
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        # adj (batch_size, K, K): adjacency matrix
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            #adj = adj.unsqueeze(1)
            #adj = torch.cat((adj,adj,adj),1)
            adj_list = [adj,adj,adj,adj]
        else:
            adj = graph.float()
            adj_list = [adj[:,1,:],adj[:,1,:],adj[:,4,:],adj[:,4,:]]
        #print(adj)

        g_feature = \
            tuple([l(graph_nodes,x) for l, x in zip(self.graph,adj_list)])
        #g_feature_0 = self.Graph_0(graph_nodes,adj[0])
        #g_feature_1 = self.Graph_1(graph_nodes,adj[1])
        #g_feature_2 = self.Graph_2(graph_nodes,adj[2])
        #g_feature_3 = self.Graph_3(graph_nodes,adj[3])
        #print('g_feature')
        #print(type(g_feature))


        g_feature = self.norm(torch.cat(g_feature,2)) + graph_nodes
        #print('g_feature')
        #print(g_feature.shape)

        graph_encode_features = self.feed_foward(g_feature) + g_feature

        return adj, graph_encode_features


# GCN
class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# Graph_Conv
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.shape)
        #print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        # print(adj.shape)
        # print(support.shape)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MVS2TEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, embedding_weight=None):
        super(MVS2TEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if embedding_weight is not None:
            if isinstance(embedding_weight, torch.Tensor):
                self.embedding.weight.data.copy_(embedding_weight)
                # self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:] # B x H
        problem_output_v1 = pade_hidden[-1,:,:] + pade_hidden[-2,:,:]  # n_layer * n_direction x B x H
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output, problem_output_v1


class MVG2TEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, embedding_weight=None):
        super(MVG2TEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        if embedding_weight is not None:
            if isinstance(embedding_weight, torch.Tensor):
                self.embedding.weight.data.copy_(embedding_weight)
                # self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        problem_output_v1 = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output, problem_output_v1


