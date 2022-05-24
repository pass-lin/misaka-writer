0# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:06:31 2021

@author: Administrator
"""

from sznlp.backend import keras,tf,K
from sznlp.backend import *
class LayerNormalization(keras.layers.Layer):
    """(Conditional) Layer Normalization
    代码来自苏剑林bert4keras
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = keras.activations.get(hidden_activation)
        self.hidden_initializer = keras.initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12
        
    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

   # @recompute_grad
    @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta
        if self.center==False and self.scale==False:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': keras.activations.serialize(self.hidden_activation),
            'hidden_initializer':
                keras.initializers.serialize(self.hidden_initializer),
            
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PositionEmbedding(keras.layers.Layer):
    """定义可训练的位置Embedding
    代码来自bert4keras
    """
    def __init__(
        self,
        max_len=None,
        output_dim=None,
        merge_mode='add',
        hierarchical=None,
        embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        custom_position_ids=False,
        initial_by_Sinusoidal=False,
        **kwargs
    ):
        '''
        max_len：输入的最大长度
        output_dim：输出维度
        merge_mode：连接的方法，默认是add，还可以"zero"返回embeding,"mul"使用乘性,
        hierarchical：是否使用层次位置
        embeddings_initializer：初始化方法,
        custom_position_ids：是否使用自定义的位置,
        '''
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids
        self.initial_by_Sinusoidal=initial_by_Sinusoidal
    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        if self.initial_by_Sinusoidal:
            seq_len = self.max_len
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]
            indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
            indices = K.pow(10000.0, -2 * indices / self.output_dim)
            embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
            embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
            embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))
            self.embeddings = self.add_weight(
                name='embeddings',
                shape=(self.max_len, self.output_dim),
                initializer=keras.initializers.constant(embeddings*0.05)
            )

        else:
            self.embeddings = self.add_weight(
                name='embeddings',
                shape=(self.max_len, self.output_dim),
                initializer=self.embeddings_initializer
            )
    @recompute_grad
    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.max_len)
            embeddings_y = K.gather(embeddings, position_ids % self.max_len)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            raise("input must is add or mul or zero")

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)
    def get_config(self):
        config = {
            'max_len': self.max_len,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                keras.initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
            'initial_by_Sinusoidal':self.initial_by_Sinusoidal,
        }
        base_config = super(PositionEmbedding, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
class SinusoidalPositionEmbedding(keras.layers.Layer):
    """定义Sin-Cos位置Embedding
    代码来自bert4keras
    """
    def __init__(
        self, output_dim=None, merge_mode='low_add',custom_position_ids=False, **kwargs
    ):
        
        #output_dim：输出维度
        #merge_mode：连接的方法，默认是add，还可以"zero"返回embeding,"mul"使用乘性,
        #custom_position_ids：是否使用自定义的位置,
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids
    @recompute_grad
    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        elif self.merge_mode == 'low_add':
            return inputs + 0.05*embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim=None,
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                 use_bias=True,
                 activation='relu',
                 bias_initializer='zeros', 
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        '''
         model_dim:全连接层的维度
         kernel_initializer：核函数初始化,
         use_bias：是否使用偏置,
        '''
        super(PositionWiseFFN,self).__init__(**kwargs)
        self.use_bias=use_bias
        self.kernel_initializer=keras.initializers.get(kernel_initializer)
        self.model_dim=model_dim
        self.activation=keras.layers.Activation(activation)
        
        self.bias_initializer=keras.initializers.get(bias_initializer)
        self.kernel_regularizer=keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer=keras.regularizers.get(bias_regularizer)
        self.activity_regularizer=keras.regularizers.get(activity_regularizer)
        self.kernel_constraint=keras.constraints.get(kernel_constraint)
        self.bias_constraint=keras.constraints.get(bias_constraint)
    def get_dense(self,dims,activations,name):
        dense=keras.layers.Dense(dims, activation=activations,use_bias=self.use_bias,kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    activity_regularizer=self.activity_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint,
                                    name=name,)
        return dense
    def define_weights(self, input_shape):
        self.l = self.get_dense(self.model_dim,self.activation,name='l_dense')
        self.o = keras.layers.Dense(input_shape[-1],None,name='o_dense')
    def build(self, input_shape):
        super(PositionWiseFFN, self).build(input_shape)  
        self.define_weights( input_shape)
    @recompute_grad
    def call(self, x):
        o = self.l(x)
        o = self.o(o)
        return o         # [n, step, dim]
    def get_config(self):
        config = {
            'model_dim': self.model_dim,
            'use_bias': self.use_bias,
            'activation':keras.activations.serialize(self.activation),
            'kernel_initializer':keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':keras.initializers.serialize(self.bias_initializer), 
            'kernel_regularizer':keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(PositionWiseFFN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape
class FFN_gate(PositionWiseFFN):
    def __init__(self,n_head=1, #头的数量
                 head_dim=None,
                 **kwargs):
        super(FFN_gate, self).__init__(**kwargs)      
        self.n_head=n_head
        self.head_dim=head_dim
    def get_config(self):
        config = {
            'n_head':self.n_head,
            'head_dim':self.head_dim,
        }
        base_config = super(FFN_gate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def define_weights(self, input_shape):
        super(FFN_gate, self).define_weights(input_shape)  
        self.local = self.get_dense(self.model_dim//self.n_head,None,name='local')
        #self.switch = self.get_dense(input_shape[-1],None)
    def local_ffn(self,x):
        batch,length=K.shape(x)[0],K.shape(x)[-2]
        x=K.reshape(x,[batch,length,self.n_head,self.head_dim])
        x=self.local(x)
        x=K.reshape(x,[batch,length,-1])
        return x
    @recompute_grad
    def call(self, x):
        o1=self.local_ffn(x)
        o2 = self.l(x)#super(FFN_gate, self).call(x)  
        o=keras.activations.sigmoid(o1)*o2
        #o=keras.activations.swish(o1)*keras.activations.swish(o2)
        return self.o(o)  
        
