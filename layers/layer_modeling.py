import torch
import torch.nn as nn
from .adapter_utils import Activations

class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)

        self.down_sampler.apply(self.init_bert_weights)
        self.up_sampler.apply(self.init_bert_weights)

        if config.use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        self.residual_before_ln = config.residual_before_ln
        self.query_before_ln = config.query_before_ln

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.hidden_size)
            self.pre_layer_norm.apply(self.init_bert_weights)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.hidden_size)
            self.post_layer_norm.apply(self.init_bert_weights)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, hidden_states, input_tensor):
        '''

        :param hidden_states: The hidden states outputted by ffn w/add & norm
        :param residual_input: The hidden states outputted by ff w/o add & norm
        :param input_tensor : residual connnection before the Transformer FFN layer
        :return:
        '''

        # preforward
        query = None

        if self.residual_before_ln:
            residual_input = hidden_states

        if self.query_before_ln:
            query = hidden_states

        if self.add_layer_norm_before_adapter:
                hidden_states = self.layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor

        if not self.residual_before_ln:
            residual_input = hidden_states

        if not self.query_before_ln:
            query = hidden_states

        # forward
        z = self.pre_layer_norm(hidden_states)
        z = self.down_sampler(z)
        z = self.activation(z)
        up = self.up_sampler(z)

        output = up
        if self.gate is not None:
            output = output * self.gate


        if self.add_layer_norm_after_adapter:
            output = self.post_layer_norm(output)

        output = output + residual_input

        return output, query, residual_input, up



class AdapterFusion(nn.Module):
    '''
    Implementation of an AdapterFusion block
    '''
    def __init__(self, adapter_config, dense_size, attention_probs_dropout_prob):
        super(AdapterFusion, self).__init__()
        self.config = adapter_config

        self.dense_size = dense_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if not self.config.query and not self.config.key and not self.config.value:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config.query:
            self.query = nn.Linear(self.dense_size, self.dense_size)
            self.query.apply(self.init_bert_weights)

        if self.config.key:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(self.init_bert_weights)

        if self.config.value:
            self.value = nn.Linear(self.dense_size, self.dense_size, bias=False)
            self.value.apply(self.init_bert_weights)
            if self.config.value_initialized:
                value_data = torch.randn(self.dense_size, self.dense_size)
                value_data[range(self.dense_size), range(self.dense_size)] = 1
                self.value.weight.data = value_data

        if self.config.temperature:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0
    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, query, key, value, residual):

        if self.config.query:
            query_layer = self.query(query)
        else:
            query_layer = query # 64, 103, 768

        if self.config.key:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config.value and self.config.value_before_softmax:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2) # batch, max_len, max_len
        attention_scores = self.dropout(attention_scores) # 64,111,3,768

        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T) # 64,111,3
        self.T = max(self.T - self.reduction, 1.0)


        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2) # 64,111,768,3

        if self.config.value and not self.config.value_before_softmax:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer
        context_layer += residual

        return attention_probs, context_layer

class AdapterFusionConfig(nn.Module):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = True

class OutputAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, output_dim):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = 16
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, output_dim)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output

    def resize_up_sampler(self, resized_size):
        self.up_sampler = nn.Linear(self.down_sample_size, resized_size)


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division,
                                      learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm,
                                      shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank,
                                      phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim,
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)



