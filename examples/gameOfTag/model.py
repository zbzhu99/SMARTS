# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from xt.model.model_utils import build_conv_layers, bulid_mlp_layers
from xt.model.ppo.ppo_cnn import PpoCnn
from xt.model.tf_compat import Input, Flatten, Model, Dense
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.layers import concatenate, Activation, Concatenate
from tensorflow.keras.layers import GRU, Reshape, Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import MSE
from tensorflow.compat.v1 import global_variables_initializer
from tensorflow.compat.v1.train import AdamOptimizer, Saver
from tensorflow.compat.v1.summary import scalar as summary_scalar
from tensorflow.compat.v1.train import linear_cosine_decay, piecewise_constant


class Tag():
    """Build Tag PPO CNN network."""

    def __init__(self, config):
        if 'filter_arches' in config['model_para']:
            self.filter_arches = config['model_para']['filter_arches']
        else:
            raise Exception(f"Expected CNN layer specification 'filter_arches' entry in 'model_para'.")

        # Verify action type used
        if config['env_para']['action_type'] == "Categorical" and config['env_para']['controller'] == "Continuous":
            if config['model_para']['action_dim'] != 5:
                raise Exception(f"Expected action_dim=5.")
        if config['env_para']['action_type'] == "Categorical" and config['env_para']['controller'] == "LaneWithContinuousSpeed":
            if config['model_para']['action_dim'] != 12:
                raise Exception(f"Expected action_dim=12.")
        elif config['env_para']['action_type'] == "DiagGaussian":
            if config['model_para']['action_dim'] != 3:
                raise Exception(f"Expected action_dim=3.")
        else:
            raise Exception(f"Unsupported action_type.")

        self._model = get_cnn_backbone(
                state_dim=self.state_dim,
                act_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                filter_arches=self.filter_arches,
                summary=self.verbose,
                action_type=config['model_para']['action_type'])

    @property
    def model(self):
        return self._model

def get_cnn_backbone(state_dim, act_dim, hidden_sizes, activation, filter_arches, summary, action_type):
    """Get CNN backbone."""
    state_input = Input(shape=state_dim, name='obs')
    conv_layer = build_conv_layers(state_input, filter_arches, activation, 'shared')
    flatten_layer = Flatten()(conv_layer)
    dense_layer = bulid_mlp_layers(flatten_layer, hidden_sizes, activation, 'shared')
    if action_type == "Categorical":
        pi_latent = Dense(act_dim, activation=None, name='pi_latent')(dense_layer)
    elif action_type == "DiagGaussian":
        pi_latent = Dense(act_dim, activation='tanh', name='pi_latent')(dense_layer)
    else:
        raise Exception(f"Error: Unsupported action_type {action_type}.")    
    out_value = Dense(1, activation=None, name='output_value')(dense_layer)
    model = Model(inputs=[state_input], outputs=[pi_latent, out_value])
    if summary:
        model.summary()

    return model

def bulid_mlp_layers(input_layer, hidden_sizes, activation, prefix=''):
    output_layer = input_layer
    for i, hidden_size in enumerate(hidden_sizes):
        output_layer = \
            Dense(hidden_size, activation=activation, name='{}_hidden_mlp_{}'.format(prefix, i))(output_layer)
    return output_layer

def build_conv_layers(input_layer, filter_arches, activation, prefix=''):
    conv_layer = input_layer
    for i, filter_arch in enumerate(filter_arches):
        filters, kernel_size, strides = filter_arch
        conv_layer = Conv2D(filters, kernel_size, strides, activation=activation, padding='valid',
                            name="{}_conv_layer_{}".format(prefix, i))(conv_layer)
    return conv_layer
