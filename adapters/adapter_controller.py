"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from adapters.adapter_utils import get_activation
from .adapter_modeling import Adapter, AdapterFusion

class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config, place="upper"):
        super().__init__()
        # low-rank adapters.
        #self.low_rank_adapters = config.low_rank_adapters
        # self.intrinsic_projections_path = os.path.join(config.output_dir, "intrinsic_projections")
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.use_fusion = config.use_fusion
        # self.device = config.device

        self.use_single_adapter = config.use_single_adapter
        self.adapters = self.construct_adapters(self.tasks)

        if place != "under" and self.use_fusion:
            self.adapter_fusion_layer = AdapterFusion(self.config, self.config.hidden_size, self.config.attention_probs_dropout_prob)
        else:
            self.adapter_fusion_layer = None

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_task(self, task):
        return task

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        if self.use_single_adapter:
            adapter = Adapter(self.config)
            for task in tasks:
                self.adapters[task] = adapter

        else:
            for task in tasks:
                self.adapters[task] = Adapter(self.config)

        return self.adapters



    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:

            adapter = self.get_adapter(task)
            for param in adapter.parameters():

                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for name, param in adapter.named_parameters():
                param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]

    def forward(self, input, input_tensor, task):
        """
        Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            tasks: the name of the tasks ["VCR"], ["VCR", "IQM"]
            input: the inputs to feed in in the adapter layer.
            input_tensor : residual connnection before the Transformer FFN layer
        Returns:
            outputs of the adapter layer.
        """
        context_outputs_all = []
        up_list = []

        if self.use_fusion:
            self.disable_adapters(task)



        for ttask in task:
            ttask = self.get_task(ttask)


            if not self.use_fusion:
                # Enables the adapter layer for the given task.
                self.enable_adapters(ttask)
                # Disable other adapters.
                other_tasks = [x for x in self.tasks if x != ttask]
                if not self.use_single_adapter: # use separate adapters
                    self.disable_adapters(other_tasks)

            adapter = self.get_adapter(ttask)
            output, query, residual, up = adapter(input, input_tensor)

            if not self.use_fusion or self.adapter_fusion_layer is None:
                return output

            up_list.append(up)
        if len(up_list)>0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1,2,0,3)

            attention_probs, context_outputs = self.adapter_fusion_layer(query, up_list, up_list, residual)



        return context_outputs # batch, max_len ,hidden

class AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = Adapter(config)

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class OutputParallelAdapterLayer(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.adapter = OutputAdapter(config, output_dim)

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs
        return outputs

    def resize_output_dim(self, resized_size):
        self.adapter.resize_up_sampler(resized_size)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(resized_size)

class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

        self.track_z = config.track_z

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        return torch.nn.functional.layer_norm(inputs, (self.input_dim,),
                                              weight=layer_norm_weights.weight,
                                              bias=layer_norm_weights.bias)

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        down = F.linear(inputs, weight=adapter_weights.down.weight,
                        bias=adapter_weights.down.bias)
        middle = get_activation(self.activation_type)(down)

        if self.track_z:
            self.z = middle

        output = F.linear(middle, weight=adapter_weights.up.weight,
                          bias=adapter_weights.up.bias)
        return output

    def forward(self, inputs, adapter_weights):
        z = self.apply_layer_norm(inputs, adapter_weights.pre_norm) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs