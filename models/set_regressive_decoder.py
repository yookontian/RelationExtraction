import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention
import copy

BertLayerNorm = nn.LayerNorm


class SetRegressiveDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False,
                 use_ILP=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        if use_ILP:
            self.decoder2class = nn.Linear(config.hidden_size, num_classes)
            self.class2hidden = nn.Linear(num_classes, config.hidden_size)
        else:
            self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)
            self.class2hidden = nn.Linear(num_classes + 1, config.hidden_size)
        # self.decoder2span = nn.Linear(config.hidden_size, 4)

        # self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        #
        # self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        #
        # self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        #
        # self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        #
        # torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        # torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        # torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        # torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        # torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        # torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        # torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        # torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

        self.output_linear = nn.Linear(config.hidden_size * 4, 4, bias=False)
        torch.nn.init.orthogonal_(self.output_linear.weight, gain=1)

        # BertOutput: layer+dropout+residual+layer_norm
        config_ = copy.deepcopy(config)
        config_.intermediate_size = config_.hidden_size

        self.head_start_context = BertOutput(config_)
        self.head_end_context = BertOutput(config_)
        self.tail_start_context = BertOutput(config_)
        self.tail_end_context = BertOutput(config_)

        self.regressive_decoder = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, batch_first=True)
        torch.nn.init.orthogonal_(self.regressive_decoder.in_proj_weight, gain=1)


    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        # print("the shape of query_embed.weight: ", self.query_embed.weight.shape)
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # print("the shape of hidden_states: ", hidden_states.shape)
        # hidden_state: [bsz, num_generated_triples, hidden_size]
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()

        # print("=====================================")
        # print(f"encoder_attention_mask:\n{encoder_attention_mask}")

        for i, layer_module in enumerate(self.layers):
            # print(f"hidden_states shape:\n{hidden_states.shape}")
            # print(f"encoder_hidden_states shape:\n{encoder_hidden_states.shape}")
            # print("=====================================")

            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        class_logits = self.decoder2class(hidden_states)

        class_hidden_states = self.class2hidden(class_logits)
        # class_hidden_states = class_hidden_states.unsqueeze(-2).repeat(1, 1, encoder_hidden_states.shape[-2], 1)
        # print(f"class_hidden_states after repeat shape:\n{class_hidden_states.shape}")
        # have a binary tensor encorder_extended_attention_mask_binary, shape is the same as encoder_extended_attention_mask, and the value is True for 0, and False for 1.
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask.view(-1, encoder_hidden_states.shape[-2])
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask
        encoder_extended_attention_mask_binary = (encoder_extended_attention_mask == 0)
        # print(f"encoder_extended_attention_mask_binary: \n{encoder_extended_attention_mask_binary}")
        # print(f"encoder_extended_attention_mask_binary shape: \n{encoder_extended_attention_mask_binary.shape}")

        encoder_extended_attention_mask_binary = encoder_extended_attention_mask_binary.unsqueeze(1).repeat(1,
                                                                                                            self.num_generated_triples,
                                                                                                            1)
        # print(f"encoder_extended_attention_mask_binary shape: \n{encoder_extended_attention_mask_binary.shape}")

        encoder_extended_attention_mask_binary = encoder_extended_attention_mask_binary.view(
            bsz * self.num_generated_triples, encoder_hidden_states.shape[-2])
        # print(f"encoder_extended_attention_mask_binary shape: \n{encoder_extended_attention_mask_binary.shape}")

        # print(f"encoder_hidden_states shape:\n{encoder_hidden_states.shape}")
        encoder_hidden_states = encoder_hidden_states.repeat(self.num_generated_triples, 1, 1)
        # print(f"encoder_hidden_states shape after:\n{encoder_hidden_states.shape}")

        # print(f"encoder_extended_attention_mask_binary shape:\n{encoder_extended_attention_mask_binary.shape}")

        class_hidden_states = class_hidden_states.view(-1, class_hidden_states.shape[-1]).unsqueeze(1)
        # print(f"class_hidden_states shape after:\n{class_hidden_states.shape}")

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1]).unsqueeze(1)
        # print(f"hidden_states shape after:\n{hidden_states.shape}")

        # print(f"encoder_extended_attention_mask_binary.unsqueeze(-1) shape:\n{encoder_extended_attention_mask_binary.unsqueeze(-1).shape}")
        print(f"class_hidden_states shape:\n{class_hidden_states.shape}")
        print(f"hidden_states shape:\n{hidden_states.shape}")
        head_start_logits_mh = self.regressive_decoder(query=encoder_hidden_states,
                                                       key=self.head_start_context(class_hidden_states, hidden_states),
                                                       value=self.head_start_context(class_hidden_states, hidden_states),
                                                       attn_mask=encoder_extended_attention_mask_binary.unsqueeze(-1))
        head_start_logits_mh = head_start_logits_mh[0]
        # print(f"head_start_logits_mh shape:\n{head_start_logits_mh.shape}")

        head_end_logits_mh = self.regressive_decoder(query=head_start_logits_mh,
                                                     key=self.head_end_context(class_hidden_states, hidden_states),
                                                     value=self.head_end_context(class_hidden_states, hidden_states),
                                                     attn_mask=encoder_extended_attention_mask_binary.unsqueeze(-1))
        head_end_logits_mh = head_end_logits_mh[0]

        # print(f"head_end_logits_mh shape:\n{head_end_logits_mh.shape}")

        tail_start_logits_mh = self.regressive_decoder(query=head_end_logits_mh,
                                                       key=self.tail_start_context(class_hidden_states, hidden_states),
                                                       value=self.tail_start_context(class_hidden_states, hidden_states),
                                                       attn_mask=encoder_extended_attention_mask_binary.unsqueeze(-1))
        tail_start_logits_mh = tail_start_logits_mh[0]

        # print(f"tail_start_logits_mh shape:\n{tail_start_logits_mh.shape}")

        tail_end_logits_mh = self.regressive_decoder(query=tail_start_logits_mh,
                                                     key=self.tail_end_context(class_hidden_states, hidden_states),
                                                     value=self.tail_end_context(class_hidden_states, hidden_states),
                                                     attn_mask=encoder_extended_attention_mask_binary.unsqueeze(-1))
        tail_end_logits_mh = tail_end_logits_mh[0]

        # print(f"tail_end_logits_mh shape:\n{tail_end_logits_mh.shape}")
        # print("=============linear==================")
        # Stack the four tensors along a new dimension in-place
        input_tensor = torch.stack([head_start_logits_mh, head_end_logits_mh, tail_start_logits_mh, tail_end_logits_mh],
                                   dim=3)

        # Flatten the input tensor along the last dimension
        input_tensor_flat = input_tensor.view(input_tensor.size(0), input_tensor.size(1), -1)
        # print(f"input_tensor_flat shape:\n{input_tensor_flat.shape}")

        # Apply the linear layer
        output_flat = self.output_linear(input_tensor_flat)
        # print(f"output_flat shape:\n{output_flat.shape}")

        # Reshape the output tensor to separate it into four tensors
        output = output_flat.view(bsz, self.num_generated_triples, output_flat.size(1), 4)

        # Split the output tensor in-place into the four individual tensors
        head_start_logits = output[:, :, :, 0].clone()
        # print(f"decoder==\nhead_start_logits: \n{head_start_logits}")
        head_end_logits = output[:, :, :, 1].clone()
        tail_start_logits = output[:, :, :, 2].clone()
        tail_end_logits = output[:, :, :, 3].clone()

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