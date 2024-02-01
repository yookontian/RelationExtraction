import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention
import math

BertLayerNorm = nn.LayerNorm

# import json
# from transformers import AutoTokenizer

class SetRegressiveDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False,
                 use_ILP=False,
                 model="bert-base-cased",
                 none_class=True,
                 positional_embedding=False,
                 LSTM_on=False):
        super().__init__()

        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.position_embedding_on = positional_embedding
        if positional_embedding:
            print("using positional embedding")
            self.position_embedding = PositionalEncoding(config.hidden_size, config.hidden_dropout_prob, max_len=num_generated_triples)
        if use_ILP:
            print("not using none class")
            self.decoder2class = nn.Linear(config.hidden_size, num_classes)
            self.relation_embed = nn.Embedding(num_classes, config.hidden_size)
        else:
            if none_class:
                print("using none class")
                self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)
                self.relation_embed = nn.Embedding(num_classes + 1, config.hidden_size)
            else:
                print("not using none class")
                self.decoder2class = nn.Linear(config.hidden_size, num_classes)
                self.relation_embed = nn.Embedding(num_classes, config.hidden_size)


        torch.nn.init.orthogonal_(self.relation_embed.weight, gain=1)

        # 20240114 updated class embedding, 20240116: didn't work.
        # load the relation emebdding
        if "span" in model:
            # using torch to load the data/NYT/embeded_rel-SpanBERT.pt
            print("using SpanBERT class embedding")
            relation_embed = torch.load("data/NYT/embeded_rel-SpanBERT.pt")
        else:
            print("using BERT class embedding")
            relation_embed = torch.load("data/NYT/embeded_rel-BERT.pt")
        # the relation_embed is in shape[num_classes, hidden_size]
        # to subplace the [0:num_classes] rows of self.relation_embed
        with torch.no_grad():  # Disable gradient tracking
            self.relation_embed.weight[0:num_classes, :] = relation_embed

        # self.decoder2span = nn.Linear(config.hidden_size, 4)

        self.LSTM_on = LSTM_on
        if self.LSTM_on:
            print("using LSTM")
            self.head_start = nn.LSTM(config.hidden_size*2, 1, num_layers=1, batch_first=True)
            self.head_end = nn.LSTM(config.hidden_size*2, 1, num_layers=1, batch_first=True)
            self.tail_start = nn.LSTM(config.hidden_size*2, 1, num_layers=1, batch_first=True)
            self.tail_end = nn.LSTM(config.hidden_size*2, 1, num_layers=1, batch_first=True)


        self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_1_back = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1_back = nn.Linear(config.hidden_size, config.hidden_size)
        #
        self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_1_back = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1_back = nn.Linear(config.hidden_size, config.hidden_size)
        #
        self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_2_back = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2_back = nn.Linear(config.hidden_size, config.hidden_size)
        #
        self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_2_back = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2_back = nn.Linear(config.hidden_size, config.hidden_size)

        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        # self.head_start_metric_3_back = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3_back = nn.Linear(config.hidden_size, 1, bias=False)

        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_3_back = nn.Linear(config.hidden_size, 1, bias=False)
        # self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3_back = nn.Linear(config.hidden_size, 1, bias=False)

        self.head_start_metric_4 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_start_metric_4_back = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_4 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_4_back = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_4 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_4_back = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_4 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_4_back = nn.Linear(config.hidden_size, 1, bias=False)

        if not self.LSTM_on:
            self.head_start = nn.Linear(config.hidden_size*2, 1, bias=False)
            self.head_end = nn.Linear(config.hidden_size*2, 1, bias=False)
            self.tail_start = nn.Linear(config.hidden_size*2, 1, bias=False)
            self.tail_end = nn.Linear(config.hidden_size*2, 1, bias=False)
            # the hidden matrix and cell matrix are initialized to 0 by default.


        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)

        torch.nn.init.orthogonal_(self.head_start_metric_1_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2_back.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2_back.weight, gain=1)

        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)


        torch.nn.init.orthogonal_(self.head_end_metric_4.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_4.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_4.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_4.weight, gain=1)



        # there won't be any trainable paras in position_embedding
        # self.position_embedding = PositionalEncoding(config.hidden_size, config.hidden_dropout_prob)


        # self.transformer_decoder_back = nn.TransformerDecoder(decoder_layer=self.decoder_layer_back, num_layers=1)
        # self.output_linear = nn.Linear(config.hidden_size * 4, 4, bias=False)
        # torch.nn.init.orthogonal_(self.output_linear.weight, gain=1)

        # BertOutput: layer+dropout+residual+layer_norm
        # config_ = copy.deepcopy(config)
        # config_.intermediate_size = config_.hidden_size

        # self.regressive_decoder = nn.MultiheadAttention(embed_dim=config.hidden_size , num_heads=1, batch_first=True)
        # torch.nn.init.orthogonal_(self.regressive_decoder.in_proj_weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        # print("the shape of query_embed.weight: ", self.query_embed.weight.shape)
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # print("the shape of hidden_states: ", hidden_states.shape)
        # hidden_state: [bsz, num_generated_triples, hidden_size]
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        # print(f"self.query_embed shape: {self.query_embed.weight.shape}")

        if self.position_embedding_on:
            hidden_states = self.position_embedding(hidden_states)

        for i, layer_module in enumerate(self.layers):

            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        # print(f"hidden_states shape:\n{hidden_states.shape}")
        class_logits = self.decoder2class(hidden_states)

        class_embedding = self.relation_embed(class_logits.argmax(-1))  # [bsz, num_generated_triples, hidden_size]

        head_start_forward_logits = torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_4(class_embedding).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))
        head_start_forward_logits_ = self.head_start_metric_3(head_start_forward_logits).squeeze()
        if (len(head_start_forward_logits_.argmax(-1).shape) == 1):
            head_start_indices = torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(
                head_start_forward_logits_.argmax(-1).unsqueeze(0))
            head_start_forward = encoder_hidden_states[head_start_indices, head_start_forward_logits_.argmax(-1).unsqueeze(0)]
            # print(f'head_start_forward shape: {head_start_forward.shape}')
        else:
            head_start_indices = torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(
                head_start_forward_logits_.argmax(-1))
            head_start_forward = encoder_hidden_states[head_start_indices, head_start_forward_logits_.argmax(-1)]


        head_end_forward_logits = torch.tanh(
            (self.head_end_metric_1(hidden_states) + self.head_end_metric_4(head_start_forward)).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))
        head_end_forward_logits_ = self.head_end_metric_3(head_end_forward_logits).squeeze()
        if (len(head_end_forward_logits_.argmax(-1).shape) == 1):
            head_end_forward = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(head_end_forward_logits_.argmax(-1).unsqueeze(0)),
                head_end_forward_logits_.argmax(-1).unsqueeze(0)
            ]
        else:
            head_end_forward = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(head_end_forward_logits_.argmax(-1)), head_end_forward_logits_.argmax(-1)
            ]


        tail_start_forward_logits = torch.tanh(
            (self.tail_start_metric_1(hidden_states) + self.tail_start_metric_4(head_end_forward)).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))
        tail_start_forward_logits_ = self.tail_start_metric_3(tail_start_forward_logits).squeeze()
        if (len(tail_start_forward_logits_.argmax(-1).shape) == 1):
            tail_start_forward = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(tail_start_forward_logits_.argmax(-1).unsqueeze(0)),
                tail_start_forward_logits_.argmax(-1).unsqueeze(0)
            ]
        else:
            tail_start_forward = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(tail_start_forward_logits_.argmax(-1)), tail_start_forward_logits_.argmax(-1)
            ]



        tail_end_forward_logits = torch.tanh(
            (self.tail_end_metric_1(hidden_states) + self.tail_end_metric_4(tail_start_forward)).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))


        tail_end_back_logits = torch.tanh(
            self.tail_end_metric_1_back(hidden_states).unsqueeze(2) + self.tail_end_metric_4_back(class_embedding).unsqueeze(2) + self.tail_end_metric_2_back(
                encoder_hidden_states).unsqueeze(1))
        tail_end_back_logits_ = self.tail_end_metric_3_back(tail_end_back_logits).squeeze()
        if (len(tail_end_back_logits_.argmax(-1).shape) == 1):
            tail_end_back = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(tail_end_back_logits_.argmax(-1).unsqueeze(0)),
                tail_end_back_logits_.argmax(-1).unsqueeze(0)
            ]
        else:
            tail_end_back = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(tail_end_back_logits_.argmax(-1)), tail_end_back_logits_.argmax(-1)
            ]


        tail_start_back_logits = torch.tanh(
            (self.tail_start_metric_1_back(hidden_states) + self.tail_start_metric_4_back(tail_end_back)).unsqueeze(2) + self.tail_start_metric_2_back(
                encoder_hidden_states).unsqueeze(1))
        tail_start_back_logits_ = self.tail_start_metric_3_back(tail_start_back_logits).squeeze()
        if (len(tail_start_back_logits_.argmax(-1).shape) == 1):
            tail_start_back = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(tail_start_back_logits_.argmax(-1).unsqueeze(0)),
                tail_start_back_logits_.argmax(-1).unsqueeze(0)
            ]
        else:
            tail_start_back = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(tail_start_back_logits_.argmax(-1)), tail_start_back_logits_.argmax(-1)
            ]


        head_end_back_logits = torch.tanh(
            (self.head_end_metric_1_back(hidden_states) + self.head_end_metric_4_back(tail_start_back)).unsqueeze(2) + self.head_end_metric_2_back(
                encoder_hidden_states).unsqueeze(1))

        head_end_back_logits_ = self.head_end_metric_3_back(head_end_back_logits).squeeze()
        if (len(head_end_back_logits_.argmax(-1).shape) == 1):
            head_end_back = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(head_end_back_logits_.argmax(-1).unsqueeze(0)),
                head_end_back_logits_.argmax(-1).unsqueeze(0)
            ]
        else:
            head_end_back = encoder_hidden_states[
                torch.arange(encoder_hidden_states.size(0)).view(-1, 1).expand_as(head_end_back_logits_.argmax(-1)), head_end_back_logits_.argmax(-1)
            ]


        head_start_back_logits = torch.tanh(
            (self.head_start_metric_1_back(hidden_states) + self.head_start_metric_4_back(head_end_back)).unsqueeze(2) + self.head_start_metric_2_back(
                encoder_hidden_states).unsqueeze(1))

        """
        # before 2024/1/7, I erroneously added this line, but it works well somehow.
        # after fix17, the pre on train set is lower than before, recall slightly lower, f1 lower.
        head_start_forward_logits = torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))
        
        # 2024/1/10, fix110_2, think the above line may only works for head_start_f_logits and tail_end_b_logits
        tail_end_back_logits = torch.tanh(
            self.tail_end_metric_1_back(hidden_states).unsqueeze(2) + self.tail_end_metric_2_back(
                encoder_hidden_states).unsqueeze(1))
        # fix110_2 didn't work, the pre on train set is lower than before.
        
        # 2024/1/10, I think the line may be helpful, so I added it for other logits, to see if it works.
        # Additionally, I commented out the line for regressive_decoder (from class_embedding = to head_start_back_logits =)
        # fix110 coef0.8 didn't work, the pre on train set is lower than before.
        head_start_back_logits = torch.tanh(
            self.head_start_metric_1_back(hidden_states).unsqueeze(2) + self.head_start_metric_2_back(
                encoder_hidden_states).unsqueeze(1))

        head_end_forward_logits = torch.tanh(
            self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))
        head_end_back_logits = torch.tanh(
            self.head_end_metric_1_back(hidden_states).unsqueeze(2) + self.head_end_metric_2_back(
                encoder_hidden_states).unsqueeze(1))

        tail_start_forward_logits = torch.tanh(
            self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))
        tail_start_back_logits = torch.tanh(
            self.tail_start_metric_1_back(hidden_states).unsqueeze(2) + self.tail_start_metric_2_back(
                encoder_hidden_states).unsqueeze(1))

        tail_end_forward_logits = torch.tanh(
            self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))
        tail_end_back_logits = torch.tanh(
            self.tail_end_metric_1_back(hidden_states).unsqueeze(2) + self.tail_end_metric_2_back(
                encoder_hidden_states).unsqueeze(1))
        """
        if self.LSTM_on:
            concated_head_start = torch.cat((head_start_forward_logits, head_start_back_logits), dim=-1)
            concated_head_start = concated_head_start.view(bsz*self.num_generated_triples, -1, concated_head_start.shape[-1])
            head_start_logits = self.head_start(concated_head_start)[0].view(bsz, self.num_generated_triples, -1)

            concated_head_end = torch.cat((head_end_forward_logits, head_end_back_logits), dim=-1)
            concated_head_end = concated_head_end.view(bsz*self.num_generated_triples, -1, concated_head_end.shape[-1])
            head_end_logits = self.head_end(concated_head_end)[0].view(bsz, self.num_generated_triples, -1)

            concated_tail_start = torch.cat((tail_start_forward_logits, tail_start_back_logits), dim=-1)
            concated_tail_start = concated_tail_start.view(bsz*self.num_generated_triples, -1, concated_tail_start.shape[-1])
            tail_start_logits = self.tail_start(concated_tail_start)[0].view(bsz, self.num_generated_triples, -1)

            concated_tail_end = torch.cat((tail_end_forward_logits, tail_end_back_logits), dim=-1)
            concated_tail_end = concated_tail_end.view(bsz*self.num_generated_triples, -1, concated_tail_end.shape[-1])
            tail_end_logits = self.tail_end(concated_tail_end)[0].view(bsz, self.num_generated_triples, -1)


        else:
        # output shape: [bsz, num_generated_triples, length, 1]
            head_start_logits = self.head_start(torch.cat((head_start_forward_logits, head_start_back_logits), dim=-1)).squeeze()
            head_end_logits = self.head_end(torch.cat((head_end_forward_logits, head_end_back_logits), dim=-1)).squeeze()
            tail_start_logits = self.tail_start(torch.cat((tail_start_forward_logits, tail_start_back_logits), dim=-1)).squeeze()
            tail_end_logits = self.tail_end(torch.cat((tail_end_forward_logits, tail_end_back_logits), dim=-1)).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        # BertIntermediate: linear + gelu
        self.intermediate = BertIntermediate(config)
        # BertOutput: layer+dropout+residual+layer_norm
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        # print("the shape of hidden_states: ", hidden_states.shape)
        # print("the shape of self_attention_outputs: ", self_attention_outputs[0].shape)

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        # The mask is also transformed so that positions with a value of 1 (indicating they should be masked) are set to
        # a large negative value (-10000.0), making them effectively zero when passed through a softmax.
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        # print("the shape of cross_attention_outputs: ", cross_attention_outputs[0].shape)
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)