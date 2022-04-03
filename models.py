from dataclasses import dataclass
from typing import Optional, Tuple
from abc import *

import torch
import torch.nn as nn

import transformers
from transformers.models.bart.modeling_bart import BartClassificationHead, BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers.file_utils import ModelOutput
from torch.nn.utils import weight_norm

from utils import init_weight


@dataclass
class SentenceClassifierOutput(ModelOutput):
    # TODO: Rewrite Docs
    """
    Base class for outputs of sentence-level classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_max_tokens)`):
            Classification scores (before Sigmoid).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class EasyBart(BartForConditionalGeneration, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.num_classes = 1
        self.classification_head = nn.Linear(config.d_model, self.num_classes)
        self.prediction_module = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )

    def classify(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SentenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        device = self.model.device
        hidden_states = outputs[0] # [B, L, D]
        all_logits = self.classification_head(hidden_states).squeeze(-1) # [B, L]

        B = input_ids.size(0)
        MAX_NUM = torch.max(input_ids.eq(self.config.eos_token_id).sum(1))

        logits = torch.full((B, MAX_NUM), -1e9, dtype=torch.float).to(device) # [B, MAX_NUM]
        for i in range(B):
            _logit = all_logits[i][input_ids[i].eq(self.config.eos_token_id)]
            l = _logit.size(0)
            logits[i, 0:l] = _logit
            
        loss = None
        if labels is not None:
            assert len(input_ids) == len(labels)
            # Create one-hot vectors indicating target sentences
            one_hot = torch.zeros((B, MAX_NUM)).to(device)
            for i in range(B):
                one_hot[i,:].index_fill_(0, labels[i][labels[i] >= 0], 1.0)
            labels = one_hot.clone()

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels) # [B]

    
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SentenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )            
    
    def predict_module(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, D]
        out = self.prediction_module(hidden_states) # [B, L, 1]
        return out.squeeze(-1).mean(dim=1) # [B]

class EasyBartLinear(EasyBart):
    def __init__(self, config: BartConfig, **kwargs):
        super(EasyBartLinear, self).__init__(config, **kwargs)

        self.classification_head = BartClassificationHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=self.num_classes,
            pooler_dropout=config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)


class EasyBartLSTM(EasyBart):
    def __init__(self, config: BartConfig, **kwargs):
        super(EasyBartLSTM, self).__init__(config, **kwargs)

        self.classification_head = LSTMClassificationHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=self.num_classes,
            pooler_dropout=config.classifier_dropout,
        )
        self.classification_head.apply(init_weight)

class EasyBartTCN(EasyBart):
    def __init__(self, config: BartConfig, **kwargs):
        super(EasyBartTCN, self).__init__(config, **kwargs)
        self.classification_head = TCNClassificationHead(
            input_size=config.d_model,
            output_size=self.num_classes,
            num_channels=[100]*10,
            kernel_size=2,
            dropout=0.2
        )
        # self.classification_head.apply(init_weight)

# == Custom Heads ================================================================================================
class LSTMClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout, num_layers=1, bidirectional=False):
        super().__init__()
        self.inner_dim = 2*inner_dim if bidirectional else inner_dim

        self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=inner_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=False,
                    )
        self.dense = nn.Linear(self.inner_dim, self.inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(self.inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        out, _ = self.lstm(hidden_states)
        out = self.dense(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.out_proj(out)
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Weight normalization is implemented via a hook that recomputes the weight tensor
        # from the magnitude and direction before every forward() call.
        conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, 
            stride=stride, padding=padding, dilation=dilation))
        chomp1 = Chomp1d(padding)
        bn1   = nn.BatchNorm1d(n_outputs)
        relu1 = nn.ReLU()
        drop1 = nn.Dropout(dropout)

        conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, 
            stride=stride, padding=padding, dilation=dilation))
        chomp2 = Chomp1d(padding)
        bn2   = nn.BatchNorm1d(n_outputs)
        relu2 = nn.ReLU()
        drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            conv1, chomp1, bn1, relu1, drop1,
            conv2, chomp2, bn2, relu2, drop2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.net[0].weight.data.normal_(0, 0.01)  # conv1
        self.net[5].weight.data.normal_(0, 0.01)  # conv2
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation_size, 
                                        padding=(kernel_size-1) * dilation_size,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TCNClassificationHead(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNClassificationHead, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, L_in, C_in)"""
        inputs = inputs.transpose(1, 2).contiguous()
        y1 = self.tcn(inputs)

        y1 = y1.transpose(1, 2).contiguous()
        out = self.linear(y1)
        return out
