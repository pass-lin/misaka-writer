# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:06:17 2021

@author: Administrator
"""
from sznlp.Other_layers import *
from sznlp.backend import keras,tf,K
from sznlp.backend import *
class MultiHead(keras.layers.Layer):
    #标准多头注意力层
    def __init__(self, n_head=1, #头的数量
                 head_dim=None,#每个头的维度
                 drop_rate=0.5, #drop比例

                 use_Time_shift=False,#是否使用time-shift
                 mask_future=False,#是否mask掉未来
                 #dense层的参数，参考keras的文档https://keras.io/zh/layers/core/#dense
                 att_activation=None,
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),#bert的初始化策略
                 att_use_bias=False,
                 bias_initializer='zeros', 
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attention_scale=True,
                 **kwargs):
        super(MultiHead, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.mask=mask_future
        self.n_head = n_head
        self.drop_rate=drop_rate       
        self.use_bias=att_use_bias
        self.kernel_initializer= keras.initializers.get(kernel_initializer)
        self.use_Time_shift=use_Time_shift
        self.attention_scale=attention_scale
        self.activation=keras.layers.Activation(att_activation)
        self.bias_initializer=keras.initializers.get(bias_initializer)
        self.kernel_regularizer=keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer=keras.regularizers.get(bias_regularizer)
        self.activity_regularizer=keras.regularizers.get(activity_regularizer)
        self.kernel_constraint=keras.constraints.get(kernel_constraint)
        self.bias_constraint=keras.constraints.get(bias_constraint)
    def get_dense(self,dims):
        #获得全连接层
        dense=keras.layers.Dense(dims, activation=self.activation,use_bias=self.use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    activity_regularizer=self.activity_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
        return dense
    def switch_weights(self,dims,name,input_shape):
        
        return self.add_weight(
            name=name,
            shape=(input_shape[0][-1], dims),
            initializer=self.kernel_regularizer,
        )

    def set_main_weights(self,input_shape):
        #获取self-attention的主要权重，有的attention变种主要改这一部分的权重获取
        self.wq = self.switch_weights(self.n_head * self.head_dim,'wq',input_shape)
        self.wk = self.switch_weights(self.n_head * self.head_dim,'wk',input_shape)
        self.wv = self.switch_weights(self.n_head * self.head_dim,'wv',input_shape)    # [n, step, h*h_dim]
        self.o_dense = self.get_dense(self.n_head * self.head_dim)
        self.o_drop = keras.layers.Dropout(rate=self.drop_rate)
        if self.attention_scale==False:
            self.wq=self.wq/(self.head_dim**0.5)
            self.wk=self.wk/(self.head_dim**0.5)
    def set_other_weights(self,input_shape):
        #获得其他权重，方便attention变种进行修改
        return 0
    def define_weights(self,input_shape):
        #一个小build函数
        self.set_main_weights(input_shape)
        self.set_other_weights(input_shape)
    def build(self,input_shape):
        #调用define_weight申请权重
        self.define_weights(input_shape)
        super(MultiHead, self).build(input_shape)
    def time_shift_pad(self,x):
        #获取time-shift
        return K.temporal_padding(x,(1,0))
    def mask_martic(self,seq_len):
        #获取常规的mask矩阵
        idxs = K.arange(0, seq_len)
        mask = idxs[None, :] <= idxs[:, None]
        mask = K.cast(mask, K.floatx())
        return 1-mask
    def time_shift(self,x):
        d=x.shape[-1]
        x=K.concatenate([self.time_shift_pad(x)[:,:-1,:d//2],x[:,:,d//2:]],-1)
        return x
    def compute_qkv(self,q,k,v):
        _q =tf.matmul(q,self.wq)      # [n, q_step, h*h_dim]
        _k, _v = tf.matmul(k,self.wk), tf.matmul(v,self.wv)
        return _q,_k,_v
    def get_QKV_mat(self,q,k,v,mask):
        #获取QKV矩阵，通过改写这一层实现某些相对位置编码
        _q,_k,_v=self.compute_qkv(q,k,v)
        q = self.split_heads(_q,mask)  # [n, h, q_step, h_dim
        k, v = self.split_heads(_k,mask), self.split_heads(_v,mask)  # [n, h, step, h_dim]
        return q,k,v
    @recompute_grad
    def call(self, inputs,mask=None,**kwargs):  

        if len(inputs)==4:
            q,k,v,LM_mask=inputs
        else:
            q,k,v=inputs[0],inputs[1],inputs[2]
            LM_mask=None
        
        
        if self.use_Time_shift:
            q=self.time_shift(q)
            k=self.time_shift(k)
            v=self.time_shift(v)
        _q, _k, _v=self.get_QKV_mat(q,k,v,mask)
        context = self.scaled_dot_product_attention(_q, _k, _v,mask=mask,LM_mask=LM_mask,inputs=[q,k,v])     # [n, q_step, h*dv]
        context=self.recover_heads(context)
        o = self.o_dense(context)       # [n, step, dim]
        o = self.o_drop(o)
        return o

    def split_heads(self, x,mask):
        #多头的情况
        x =K.reshape(x, (-1, K.shape(x)[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])       # [n, h, step, h_dim]
    def split_heads_2(self, x,mask):
        x =K.reshape(x, (-1, K.shape(x)[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3]) 
    def recover_heads(self,context):
        
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = K.reshape(context, (K.shape(context)[0], K.shape(context)[1],self.n_head * self.head_dim))
        return context
    def pay_QK(self,q,k,inputs):
        return tf.matmul(q, k, transpose_b=True) 
    def pay_attention_V(self,attention,v,inputs):
        return tf.matmul(attention, v)  
    def divide_dk(self,score,q,k):
        if self.attention_scale:
            dk = K.cast(K.shape(score)[-1], dtype=tf.float32)
            score=score/ (tf.math.sqrt(dk) + 1e-8)
        return score
    def apply_attention(self,score,v,mask,inputs):
        
        q_mask, v_mask = None, None
        
        if mask!=None:
            q_mask, v_mask=mask[0],mask[1]
            if v_mask!=None:
                v_mask=tf.cast(v_mask,score.dtype)
                v_mask=1-tf.reshape(v_mask,[-1,1,1,K.shape(v_mask)[-1]])
                score+=v_mask*-1e9
    
        self.attention = keras.activations.softmax(score, axis=-1)                               # [n, h, q_step, step]
        context = self.pay_attention_V(self.attention, v,inputs)       # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        return context
    def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        
        score =self.pay_QK(q, k,inputs)
        score=self.divide_dk(score,q,k)
         # [n, h_dim, q_step, step]
        if LM_mask!=None:
            score += LM_mask * -1e9
        elif self.mask==True:
            #mask
            LM_mask=self.mask_martic(K.shape(score)[-1])
            score += LM_mask * -1e9
        
        return self.apply_attention(score, v, mask,inputs)
    def get_config(self):
        config = {
            'n_head': self.n_head,
            'head_dim': self.head_dim,
            'drop_rate': self.drop_rate,
            'att_use_bias': self.use_bias,
            'use_Time_shift':self.use_Time_shift,
            'mask_future':self.mask,
            'attention_scale':self.attention_scale,
            'att_activation':keras.activations.serialize(self.activation),
            'kernel_initializer':keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':keras.initializers.serialize(self.bias_initializer), 
            'kernel_regularizer':keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(MultiHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape
def Roformer(attention):
    class roformer(attention):
        def roform_position(self,x):
            batch_size,n_head,sequence_length,diemension=K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],K.shape(x)[3]
            d=K.cast(diemension,dtype=K.floatx())
            position_ids = K.arange(0, sequence_length, dtype=K.floatx())[None]
            indices = K.arange(0, diemension, dtype=K.floatx())
            indices = K.pow(10000.0, -2 * indices / d)
            indices = tf.einsum('bn,d->bnd', position_ids, indices)
            sin=K.sin(indices)
            cos=K.cos(indices)
            x_=K.stack([-1*x[...,1::2],x[...,::2]],4)
            x_=K.reshape(x_,[batch_size,n_head,sequence_length,diemension])
            return cos*x+sin*x_
        def get_QKV_mat(self,q,k,v,mask):
            q,k,v=super(roformer, self).get_QKV_mat(q,k,v,mask)
            q,k=self.roform_position(q),self.roform_position(k)
            return q,k,v
    return roformer
class AFT_full(MultiHead):
    def __init__(self,max_len=None,#最大长度
                 **kwargs):
        super(AFT_full, self).__init__(**kwargs)      
        self.max_len=max_len
    def set_other_weights(self, input_shape):
        self.bias=self.add_weight(name='bias', 
                                shape=(self.max_len,self.max_len),
                                initializer=self.kernel_initializer,
                                trainable=True)
    def mask_pad(self,w,mask=None,LM_mask=None):
        
        if LM_mask!=None:
            w +=LM_mask * -1e9
        elif self.mask==True:
            LM_mask=self.mask_martic(K.shape(w)[-1])
            w+=LM_mask*-1e9
        q_mask, v_mask = None, None
        if mask!=None:
            
            q_mask, v_mask=mask[0],mask[1] 
            
            v_mask=K.cast(v_mask,w.dtype)
            v_mask=1-K.reshape(v_mask,[-1,1,1,K.shape(v_mask)[-1]])
            w+=v_mask*-1e9
        w=K.exp(w)
        score=w
        
        return w
    def get_bias(self,q,k,**kwargs):
        seq_len_1 = K.shape(k)[-2]
        seq_len_2 = K.shape(q)[-2]
        w=self.bias[:seq_len_2,:seq_len_1]
        return w
    def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        w=self.get_bias(q,k)
        k=K.exp(k)
        w=self.mask_pad(w,mask,LM_mask)
        
        temp =  w@ (k* v)
        weighted = temp / (w @ k)
        return keras.activations.sigmoid(q)*weighted
    def get_config(self):
        config = {
            'max_len':self.max_len,
            
        }
        base_config = super(AFT_full, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Synthesizer_R(MultiHead):
    #《Synthesizer: Rethinking Self-Attention in Transformer Models》中的R实现
    #也可以用作MLP-Mixer
     def __init__(self,train_able=True,max_len=256,**kwargs):
         #train_able指的是R矩阵是被训练
         super(Synthesizer_R, self).__init__(**kwargs)
         self.train_able=train_able
         self.max_len=max_len
     def set_main_weights(self,input_shape):
        self.wv = self.switch_weights(self.n_head * self.head_dim,'wv',input_shape)
        self.R=self.bias=self.add_weight(name='random_bias', 
                                shape=(self.max_len,self.max_len),
                                initializer=self.kernel_initializer,
                                trainable=self.train_able)
        self.o_dense = keras.layers.Dense(self.n_head * self.head_dim,use_bias=self.use_bias,kernel_initializer=self. kernel_initializer)
        self.o_drop = keras.layers.Dropout(rate=self.drop_rate)
        if self.attention_scale==False:
            self.wq=self.wq/(self.head_dim**0.5)
            self.wk=self.wk/(self.head_dim**0.5)        
     def compute_qkv(self,q,k,v):
        v=tf.matmul(v,self.wv)
        return q,k,v
     def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        score=self.R[:K.shape(v)[-2],:K.shape(v)[-2]]
         # [n, h_dim, q_step, step]
        score=self.divide_dk(score,q,k)
        return self.apply_attention(score, v, mask,inputs)
     def get_config(self):
        config = {
            'train_able': self.train_able,
            'max_len':self.max_len,
        }
        base_config = super(Synthesizer_R, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
MLP_Mixer=Synthesizer_R


class Distance_mat():
    def __init__(self):
        self.length=-1
        self.mat=None
    def distance_martic(self,length,head_dim,beta,dims,n_head):
            head=(K.arange(0,n_head,dtype=K.floatx())+1)/n_head*beta
            #head=K.reshape(K.cast(beta,'float32'),[1])
            arange=K.expand_dims(K.arange(1,length+1,dtype=K.floatx()))
            dims_arragne=K.expand_dims(K.arange(1,dims+1,dtype=K.floatx()),0)/dims+0.1
            dims_arragne=K.repeat(dims_arragne,length)
            arange=arange*dims_arragne
            
            mat=tf.einsum('i,ajk->ijk', head, arange)
            return mat
    def get_mat(self,length,head_dim,beta,dims,n_head):
        if self.length==-1:
            self.mat=self.distance_martic(length,head_dim,beta,dims,n_head)
        elif length!=self.length:
            self.mat=self.distance_martic(length,head_dim,beta,dims,n_head)
            self.length=length
        return self.mat
distance_mat=None


def Distance(attention:MultiHead):
    #只考虑距离的位置编码
    global  distance_mat
    if distance_mat==None:
        distance_mat=Distance_mat()
    class distance(attention):
        def __init__(self,beta=1/500,**kwargs):
            super(distance, self).__init__(**kwargs)
            self.beta=beta
        def get_QKV_mat(self,q,k,v,mask):
        #获取QKV矩阵，通过改写这一层实现某些相对位置编码
            _q =tf.matmul(q,self.wq)      # [n, q_step, h*h_dim]
            _k, _v = tf.matmul(k,self.wk), tf.matmul(v,self.wv)
            q = self.split_heads_2(_q,mask,0)  # [n, h, q_step, h_dim
            k, v = self.split_heads_2(_k,mask,0), self.split_heads(_v,mask) #k的flag是1代表bias要*-1

            # [n, h, step, h_dim]
            return q,k,v
        def compute_distance(self,x,bias):
            x=tf.multiply(bias,x)
            x=tf.cumsum(x,-2)/tf.cumsum(bias,-2)
            
            return x
        def distance(self,x,flag):
            length=K.shape(x)[-2]
            bias=distance_mat.get_mat(length,self.head_dim,self.beta,
                                      self.head_dim,self.n_head)    
            if flag:
                bias=bias*-1
            bias=K.exp(bias)
            return self.compute_distance(x,bias)
        def split_heads_2(self, x,mask,flag):
            x=super(distance, self).split_heads(x,mask)
            return self.distance(x,flag)
        def get_config(self):
            config = {
                'beta':self.beta,
            }
            base_config = super(distance, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    return distance
def Distance_train(attention:MultiHead):
    #只考虑距离的位置编码
    
    class distance(attention):
        def __init__(self,max_len=100,beta=1.05,**kwargs):
            super(distance, self).__init__(**kwargs)
            self.max_len=max_len
            self.beta=beta
        def distance_martic(self,dims,beta,length):
            head=(K.arange(0,self.n_head,dtype=K.floatx())+1)/self.n_head*beta/length
            arange=K.expand_dims(K.arange(1,length+1,dtype=K.floatx()))
            dims_arragne=K.expand_dims(K.arange(1,dims+1,dtype=K.floatx()),0)/dims
            dims_arragne=K.repeat(dims_arragne,length)
            arange=K.pow(arange,dims_arragne)
            mat=tf.einsum('i,ajk->ijk', head, arange)
            return mat
        def get_QKV_mat(self,q,k,v,mask):
        #获取QKV矩阵，通过改写这一层实现某些相对位置编码
            _q =tf.matmul(q,self.wq)      # [n, q_step, h*h_dim]
            _k, _v = tf.matmul(k,self.wk), tf.matmul(v,self.wv)
            q = self.split_heads_2(_q,mask,self.alphaq)  # [n, h, q_step, h_dim
            k, v = self.split_heads_2(_k,mask,self.alphak), self.split_heads(_v,mask)  # [n, h, step, h_dim]
            return q,k,v
        def get_bias(self,name):
            
            bias=self.add_weight(name=name, 
                                shape=(self.n_head,self.max_len,self.head_dim),
                                initializer=keras.initializers.constant(self.mat),
                                trainable=True)
            return bias
        def initial_bias(self):
            self.mat=Distance_mat().get_mat(self.max_len,self.head_dim,self.beta/self.max_len,
                                      self.head_dim,self.n_head)
        def set_other_weights(self,input_shape):
            super(distance,self).set_other_weights(input_shape)
            self.initial_bias()
            self.alphaq=self.get_bias('bias_q')
            self.alphak=self.get_bias('bias_k')
        def compute_distance(self,x,bias):
            x=tf.multiply(bias,x)
            x=tf.cumsum(x,-2)/tf.cumsum(bias,-2)
            return x
        def get_distance_bias(self,x,alpha):
            length=K.shape(x)[-2]
            bias=alpha[:,:length,:]#self.mat[:,:length,:]
            return K.exp(bias) 
        def distance(self,x,alpha):
            
            
            bias=self.get_distance_bias(x,alpha)      
            return self.compute_distance(x,bias)
        def split_heads_2(self, x,mask,alpha):
            x=super(distance, self).split_heads(x,mask)
            return self.distance(x,alpha) 
        def get_config(self):
            config = {
                'max_len':self.max_len,
                'beta':self.beta,
            }
            base_config = super(distance, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    return distance
def sumformer(attention:MultiHead):
    attention=Distance_train(attention)
    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.005)
    def random_init(shape, dtype=None):
        bias=kernel_initializer(shape)
        bias=K.abs(bias)
        bias=tf.cumsum(bias,-1)
        return bias
    class sumformer(attention):
        def __init__(self,**kwargs):
            super(sumformer, self).__init__(**kwargs)
            
        def get_bias(self,name):
            
            bias=self.add_weight(name=name, 
                                shape=(self.n_head,self.max_len,self.head_dim),
                                initializer=random_init,
                                trainable=True)
            return bias

    return sumformer
class RelativePositionEmbedding(keras.layers.Layer):
    """相对位置编码 来自bert4keras"""
    def __init__(
        self, max_len=None, 
        output_dim=None,
        embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        **kwargs
    ):
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.max_len= max_len+1
        self.output_dim = output_dim
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='relative',
            shape=(self.max_len, self.output_dim),
            initializer=self.embeddings_initializer,
        )
        self.index=self.compute_position_ids(self.max_len*2,self.max_len*2)
    @recompute_grad
    def call(self, inputs):
        q,v=inputs
        q_len=K.shape(q)[-2]
        v_len=K.shape(v)[-2]
        if q_len<self.max_len*2 and v_len<self.max_len*2:
            pos_ids =self.index[:q_len,:v_len]
        else:
            pos_ids = self.compute_position_ids(q_len,v_len)
        return K.gather(self.embeddings, pos_ids)

    def compute_position_ids(self, q_len,v_len):
        # 计算位置差
        q_idxs = K.arange(0, q_len, dtype='int32')
        q_idxs = K.expand_dims(q_idxs, 1)
        v_idxs = K.arange(0, v_len, dtype='int32')
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        # 后处理操作
        max_position = (self.max_len - 1) // 2
        pos_ids = K.clip(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return pos_ids+1

    def get_config(self):
        config = {
            'max_len': self.max_len,
            'output_dim': self.output_dim,
            'embeddings_initializer':
                keras.initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class RelativePositionEmbeddingT5(RelativePositionEmbedding):
    """Google T5的相对位置编码 来自bert4keras
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(
        self,
        num_buckets=None,
        output_dim=None,
        max_len=128,
        bidirectional=True,
        embeddings_initializer='zeros',
        **kwargs
    ):
        super(RelativePositionEmbeddingT5,
              self).__init__(max_len, output_dim, **kwargs)
        self.max_distance = max_len+1
        self.bidirectional = bidirectional
        self.num_buckets=num_buckets
    def compute_position_ids(self, q_len,v_len):
        """T5的相对位置分桶（直接翻译自官方T5源码）
        """
        # 计算位置差
        q_idxs = K.arange(0, q_len, dtype='int32')
        q_idxs = K.expand_dims(q_idxs, 1)
        v_idxs = K.arange(0, v_len, dtype='int32')
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        # 后处理操作
        num_buckets, max_distance = self.num_buckets, self.max_distance
        ret = 1
        n = -pos_ids+1
        if self.bidirectional:
            num_buckets //= 2
            ret += K.cast(K.less(n, 0), 'int32') * num_buckets
            n = K.abs(n)
        else:
            n = K.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = K.less(n, max_exact)
        val_if_large = max_exact + K.cast(
            K.log(K.cast(n, K.floatx()) / max_exact) /
            K.log(max_distance / max_exact) * (num_buckets - max_exact),
            'int32',
        )
        val_if_large = K.minimum(val_if_large, num_buckets - 1)
        ret += K.switch(is_small, n, val_if_large)
        return ret

    def get_config(self):
        config = {
            'max_len': self.max_distance-1,
            'bidirectional': self.bidirectional,
            'num_buckets':self.num_buckets,
        }
        base_config = super(RelativePositionEmbeddingT5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def T5_relative(attention:MultiHead):
     class t5_relative(attention):
         def __init__(self,max_len=100,num_buckets=32,**kwargs):
            super(t5_relative, self).__init__(**kwargs)
            self.max_len=max_len
            
            self.num_buckets=num_buckets
         def set_main_weights(self,input_shape):
             super(t5_relative,self).set_main_weights(input_shape)
             self.relative=RelativePositionEmbeddingT5(self.num_buckets,self.n_head,self.max_len)
         def divide_dk(self,score,q,k):
            score=super(t5_relative, self).divide_dk(score,q,k)
            position_bias = K.permute_dimensions(self.relative([q,k]), (2, 0, 1))
            score = score + K.expand_dims(position_bias, 0)
            return score
         def get_config(self):
            config = {
                'max_len':self.max_len,
                
                'num_buckets':self.num_buckets,
            }
            base_config = super(t5_relative, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
     return t5_relative
def Deberta_attention(attention:MultiHead):
    #deberta的相对位置编码
    #https://arxiv.org/abs/2006.03654
    class deberta_attention(attention):
        def __init__(self,max_len=100,**kwargs):
            super(deberta_attention, self).__init__(**kwargs)
            self.max_len=max_len
        def set_other_weights(self,input_shape):
            super(deberta_attention,self).set_other_weights(input_shape)
            self.bias=RelativePositionEmbedding(self.max_len,self.head_dim,self.kernel_initializer)
        def get_QKV_mat(self,q,k,v,mask):
        
            q_QK =tf.matmul(q,self.wq)@tf.transpose(self.wk)  # [n, q_step, h*h_dim]
            k_KQ= tf.matmul(k,self.wk)@tf.transpose(self.wq)
            _v=tf.matmul(v,self.wv)
            q_QK = self.split_heads(q_QK,mask)  # [n, h, q_step, h_dim
            k_KQ, v = self.split_heads(k_KQ,mask), self.split_heads(_v,mask)  # [n, h, step, h_dim]
            return q_QK,k_KQ,v
        def pay_QK(self,q_QK,k_KQ,inputs):
            R=self.bias([q_QK,k_KQ])
            q=self.split_heads(inputs[0],None)
            
            
            a1=q @tf.transpose(k_KQ,[0,1,3,2])
            a2=tf.einsum('bhjd,jkd->bhjk', q_QK, R)
            a3=tf.einsum('jkd,bhkd->bhjk', R, k_KQ)
            return a1+a2+a3
            
        def get_config(self):
            config = {
                'max_len':self.max_len,
            }
            base_config = super(deberta_attention, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    return deberta_attention
class LocalRNN(keras.layers.Layer):
    def __init__(self, output_dim=None, 
                 rnn_type='rnn', 
                 ksize=3,
                 dropout=0,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),#bert的初始化策略
                 bias_initializer='zeros', 
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocalRNN, self).__init__(**kwargs)
        self.output_dim=output_dim
        self.rnn_type=rnn_type
        self.ksize=ksize
        self.dropout=dropout
        self.activation=keras.layers.Activation(activation)
        self.use_bias=use_bias
        self.kernel_initializer= keras.initializers.get(kernel_initializer)
        self.bias_initializer=keras.initializers.get(bias_initializer)
        self.kernel_regularizer=keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer=keras.regularizers.get(bias_regularizer)
        self.activity_regularizer=keras.regularizers.get(activity_regularizer)
        self.kernel_constraint=keras.constraints.get(kernel_constraint)
        self.bias_constraint=keras.constraints.get(bias_constraint)

         
        

        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        self.select_index = tf.constant(idx)
    def initial_RNN(self,RNN):
        return RNN(units=self.output_dim,
                   dropout=self.dropout,
                 kernel_initializer=self.kernel_initializer,#bert的初始化策略
                 bias_initializer=self.bias_initializer, 
                 kernel_regularizer=self.kernel_regularizer,
                 bias_regularizer=self.bias_regularizer,
                 activity_regularizer=self.activity_regularizer,
                 kernel_constraint=self.kernel_constraint,
                 bias_constraint=self.bias_constraint,
                   )
    def initiali_weights(self,input_shape):
        self.input_dims=input_shape[-1]
        self.zeros = K.zeros((self.ksize-1, input_shape[-1]),dtype=K.floatx())
        if self.rnn_type.lower()=='lstm':
            RNN=keras.layers.LSTM
        elif self.rnn_type.lower()=='gru':
            RNN=keras.layers.GRU
        else:
            RNN=keras.layers.SimpleRNN
        self.rnn=self.initial_RNN(RNN)
        self.dense=keras.layers.Dense(self.output_dim, activation=self.activation,use_bias=self.use_bias,kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    activity_regularizer=self.activity_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint)
    def build(self,input_shape):
        #调用define_weight申请权重
        self.initiali_weights(input_shape)
        super(LocalRNN, self).build(input_shape)
    def get_K(self, x):
        
        batch_size,l=K.shape(x)[0], K.shape(x)[1]
        zeros = K.repeat(self.zeros,batch_size)
        zeros=K.reshape(zeros,[-1,self.ksize-1,self.input_dims])
        x = K.concatenate((zeros, x), axis=1)
        key = tf.gather(x,self.select_index[:self.ksize*l],axis=1)
        
        key = K.reshape(key,(batch_size,l, self.ksize, self.output_dim))
        return key
    @recompute_grad
    def call(self, x): 
        batch_size,l=K.shape(x)[0], K.shape(x)[1]
        x=self.dense(x)
        x = self.get_K(x) # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        h=K.reshape(x,(-1, self.ksize,self.input_dims))
        h = self.rnn(h)
        
        return K.reshape(h,[batch_size,-1,self.output_dim])
    def compute_output_shape(self,input_shape):
        
        return input_shape
    def get_config(self):
        config = {
            'output_dim':self.output_dim,
            'rnn_type':self.rnn_type,
            'ksize':self.ksize,
            'dropout':self.dropout,
            'activation':self.activation,
            'use_bias':self.use_bias,
            'kernel_initializer':keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':keras.initializers.serialize(self.bias_initializer), 
            'kernel_regularizer':keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(LocalRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class LocalRNN_layer(LocalRNN):
    def __init__(self,
            center=True,
            scale=True,
            epsilon=None,
            hidden_units=None,
            hidden_activation='linear',
            hidden_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
             **kwargs):
            
            self.center = center
            self.scale = scale
            self.hidden_units = hidden_units
            self.hidden_activation = keras.activations.get(hidden_activation)
            self.hidden_initializer = keras.initializers.get(hidden_initializer)
            self.epsilon = epsilon or 1e-12
            super(LocalRNN_layer, self).__init__(**kwargs)
    def initiali_weights(self,input_shape):
            
            self.drop = keras.layers.Dropout(rate=self.dropout)
            self.localrnn=LocalRNN( 
                output_dim=self.output_dim,
                rnn_type=self.rnn_type,
                ksize=self.ksize,
                dropout=self.dropout,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint)
            self.rnn_lm=LayerNormalization(
                center=self.center,
                scale=self.scale,
                epsilon=self.epsilon,
                hidden_units=self.hidden_units,
                hidden_activation=self.hidden_activation,
                hidden_initializer=self.hidden_initializer,)
    @recompute_grad
    def call(self,q):
        x=self.localrnn(self.drop(q))
        x=keras.layers.Add()([x,q])
        return self.rnn_lm(x)
    def get_config(self):
            config = {
                'center': self.center,
                'scale': self.scale,
                'epsilon': self.epsilon,
                'hidden_units': self.hidden_units,
                'hidden_activation': keras.activations.serialize(self.hidden_activation),
                'hidden_initializer':
                    keras.initializers.serialize(self.hidden_initializer),
            }
            base_config = super(LocalRNN_layer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
def R_Transformer(attention:MultiHead):
    class R_transfomer(attention):
        def __init__(self,
             rnn_type='gru', 
             ksize=3,
             activation='relu',
             use_bias=True,
            center=True,
            scale=True,
            epsilon=None,
            hidden_units=None,
            hidden_activation='linear',
            hidden_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            rnn_layer_num=1,
             **kwargs):
            super(R_transfomer, self).__init__(**kwargs)
            self.rnn_type=rnn_type
            self.rnn_layer_num=rnn_layer_num
            self.ksize=ksize
            self.activation=keras.layers.Activation(activation)
            self.use_bias=use_bias
            self.center = center
            self.scale = scale
            self.hidden_units = hidden_units
            self.hidden_activation = keras.activations.get(hidden_activation)
            self.hidden_initializer = keras.initializers.get(hidden_initializer)
            self.epsilon = epsilon or 1e-12
        def set_other_weights(self,input_shape):
            super(R_transfomer, self).set_other_weights(input_shape)
            self.rnn_layers=[]
            for _ in range(self.rnn_layer_num):
                layer=LocalRNN_layer( 
                    output_dim=self.head_dim,
                    rnn_type=self.rnn_type,
                    ksize=self.ksize,
                    dropout=self.drop_rate,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    center=self.center,
                    scale=self.scale,
                    epsilon=self.epsilon,
                    hidden_units=self.hidden_units,
                    hidden_activation=self.hidden_activation,
                    hidden_initializer=self.hidden_initializer)
                self.rnn_layers.append(layer)
            self.wa = self.switch_weights(self.head_dim,'wa',input_shape)
            self.wb = self.switch_weights(self.n_head * self.head_dim,'wb',[[self.head_dim]])
        def compute_qkv(self,q,k,v):
            q=q@ self.wa
            for layer in self.rnn_layers:
                q=layer(q)
            
            q=q@ self.wb
            return super(R_transfomer,self).compute_qkv(q,q,q)
        def get_config(self):
            config = {
                'rnn_type':self.rnn_type,
                'ksize':self.ksize,
                'activation':self.activation,
                'use_bias':self.use_bias,
                'center': self.center,
                'scale': self.scale,
                'epsilon': self.epsilon,
                'hidden_units': self.hidden_units,
                'hidden_activation': keras.activations.serialize(self.hidden_activation),
                'hidden_initializer':
                    keras.initializers.serialize(self.hidden_initializer),
                'rnn_layer_num':self.rnn_layer_num,
            }
            base_config = super(LocalRNN, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    return R_transfomer
class Nyströmformer(MultiHead):
    def __init__(self,mean_length=64,iters = 6,**kwargs):
        super(Nyströmformer, self).__init__(**kwargs)
        self.iters=iters
        self.mean_length=mean_length
        self.attention_scale=True
    def get_config(self):
            config = {
                'mean_length':self.mean_length,
                'iters':self.iters,
            }
            base_config = super(Nyströmformer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    def set_other_weights(self,input_shape):
        self.pool=keras.layers.AveragePooling2D(pool_size=(1, self.mean_length),padding='same')
    def moore_penrose_iter_pinv(self,x):
        from einops import rearrange

        abs_x = K.abs(x)
        col = K.sum(abs_x,-1)
        row = K.sum(abs_x,-2)
        z = rearrange(x, '... i j -> ... j i') / (K.max(col) * K.max(row))
        
        I = tf.eye(K.shape(x)[-2],dtype=K.floatx())
        I = rearrange(I, 'i j -> () i j')
        for _ in range(self.iters):
            xz = x @ z
            z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
        return z
    def get_mini_mat(self,x):
        x=self.pool(x)
        return x
    def qk_product(self,q,k,v_mask):
        score =tf.matmul(q, k, transpose_b=True) 
        v_mask=tf.cast(score==0,score.dtype)
        score+=v_mask*-1e9
        return keras.activations.softmax(score,axis=-1)
    def scaled_dot_product_attention(self, q, k, v,mask=None,inputs=None,LM_mask=None):
        if LM_mask!=None:
            raise('Nyströmformer only be used by self-attention\n'+'Nyströmformer只是能自注意力时使用')
        if self.attention_scale:
            q=q*(self.head_dim**-0.5)
        q_=self.get_mini_mat(q)
        k_=self.get_mini_mat(k)
        q_mask,v_mask=mask[0],mask[1]
        mat1=self.qk_product(q,k_,v_mask)
        mat2=self.qk_product(q_,k_,v_mask)
        mat3=self.qk_product(q_,k,v_mask)
        mat2= self.moore_penrose_iter_pinv(mat2)
        score=(mat1@mat2)@(mat3@v)
        return score
class AFT_gate(AFT_full):
    def __init__(self,window_size=32,#窗口大小
                 **kwargs):
        super(AFT_gate, self).__init__(**kwargs)      
        self.window_size=window_size
    def get_config(self):
        config = {
            'window_size':self.window_size,
            
        }
        base_config = super(AFT_gate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def split_local(self,x):
        shape=K.shape(x)
        if shape[-2]%self.window_size>0 :
            size=self.window_size-tf.shape(x)[-2]%self.window_size
            x=tf.pad(x,[[0,0],[0,0],[0,size],[0,0]])
        shape=K.shape(x)
        x=K.reshape(x,[shape[0],shape[1],shape[-2]//self.window_size,self.window_size,shape[-1]])
        return x
    def recover_local(self,x,shape):
        x=K.reshape(x,[shape[0],shape[1],-1,shape[-1]])
        if shape[-2]%self.window_size>0:
            x=x[:,:,:shape[-2],:]
        return x
    def attention_bias(self,score,q,k):
        return score
    def local_attention(self, _q, _k, _v):
        
        shape=K.shape(_q)
        
        q=self.split_local(_q)
        k=self.split_local(_k)
        v=self.split_local(_v)
        score=q@tf.transpose(k,[0,1,2,4,3])
        score=self.attention_bias(score,q,k)
        if self.mask==True:
            LM_mask=self.mask_martic(K.shape(q)[-2])
            score+=LM_mask*-1e9
        score=self.divide_dk(score,q,k)
        mask=K.cast(score==0,score.dtype)
        score=score+mask*-1e9
        score=keras.activations.softmax(score,-1)
        o=score@v
        return self.recover_local(o, shape)
    def global_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        w=self.get_bias(q,k)
        k=K.exp(k)
        w=self.mask_pad(w,mask,LM_mask)
        temp =  w@ (k* v)
        weighted = temp / (w @ k)
        return weighted
    def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        o1=self.local_attention( q, k, v)
        o2=self.global_attention( q, k, v,compute_mask,mask,inputs,LM_mask)
        return keras.activations.sigmoid(o1)*o2
class gate_attention(AFT_gate):
    def set_other_weights(self, input_shape):
        #self.relative=RelativePositionEmbedding(max_len=self.max_len, output_dim=self.n_head)
        self.relative=RelativePositionEmbeddingT5(num_buckets=32,max_len=self.max_len, output_dim=self.n_head,embeddings_initializer=self.kernel_initializer)
    def attention_bias(self,score,q,k):
        position_bias=self.relative([q,k])
        position_bias = K.permute_dimensions(position_bias, (2, 0, 1))
        w = K.reshape(position_bias, [1,self.n_head,1,self.window_size,self.window_size])
        return score+w
    def get_bias(self,q,k):
        position_bias = K.permute_dimensions(self.relative([q,k]), (2, 0, 1))
        w = K.expand_dims(position_bias, 0)
        return w
    def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        o1=self.local_attention( q, k, v)
        o2=self.global_attention( q, k, v,compute_mask,mask,inputs,LM_mask)
        return keras.activations.sigmoid(o1)*o2
class AFT_relative(AFT_full):
    def set_other_weights(self, input_shape):
        self.relative=RelativePositionEmbedding(max_len=self.max_len, output_dim=self.n_head)
    def get_bias(self,q,k):
        position_bias = K.permute_dimensions(self.relative([q,k]), (2, 0, 1))
        w = K.expand_dims(position_bias, 0)
        return w

class gate_attention_tiny(AFT_gate):
    def global_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        w=LM_mask
        k=K.exp(k)
        temp =  w@ (k* v)
        weighted = temp / (w @ k)
        return weighted
    def set_other_weights(self, input_shape):
        #self.relative=RelativePositionEmbedding(max_len=self.max_len, output_dim=self.n_head)
        self.relative=RelativePositionEmbeddingT5(num_buckets=32,
                                                  max_len=self.window_size,
                                                  output_dim=self.n_head,
                                                  embeddings_initializer=self.kernel_initializer)

    def attention_bias(self,score,q,k):
        position_bias=self.relative([q,k])
        position_bias = K.permute_dimensions(position_bias, (2, 0, 1))
        w = K.reshape(position_bias, [1,self.n_head,1,self.window_size,self.window_size])
        return score+w

    def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        o1=self.local_attention( q, k, v)
        o2=self.global_attention( q, k, v,compute_mask,mask,inputs,LM_mask)
        return keras.activations.sigmoid(o1)*o2

class gate_attention_tiny_one_head(AFT_gate):
    def global_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        w=LM_mask
        k=K.exp(k)
        temp =  w@ (k* v)
        weighted = temp / (w @ k)
        return weighted
    def set_other_weights(self, input_shape):
        #self.relative=RelativePositionEmbedding(max_len=self.max_len, output_dim=self.n_head)
        self.relative=RelativePositionEmbeddingT5(num_buckets=32,
                                                  max_len=self.window_size,
                                                  output_dim=1,
                                                  embeddings_initializer=self.kernel_initializer)
    def set_main_weights(self,input_shape):
        #获取self-attention的主要权重，有的attention变种主要改这一部分的权重获取
        self.wq = self.switch_weights(self.head_dim,'wq',input_shape)
        self.wk = self.switch_weights(self.n_head * self.head_dim,'wk',input_shape)
        self.wv = self.switch_weights(self.n_head * self.head_dim,'wv',input_shape)    # [n, step, h*h_dim]
        self.o_dense = self.get_dense(self.n_head * self.head_dim)
        self.o_drop = keras.layers.Dropout(rate=self.drop_rate)
        if self.attention_scale==False:
            self.wq=self.wq/(self.head_dim**0.5)
            self.wk=self.wk/(self.head_dim**0.5)
    def attention_bias(self,score,q,k):
        position_bias=self.relative([q,k])
        position_bias = K.permute_dimensions(position_bias, (2, 0, 1))
        w = K.reshape(position_bias, [1,1,1,self.window_size,self.window_size])
        return score+w
    def get_QKV_mat(self,q,k,v,mask):
        #获取QKV矩阵，通过改写这一层实现某些相对位置编码
        _q,_k,_v=self.compute_qkv(q,k,v)
        q = K.expand_dims(_q,1)  # [n, h, q_step, h_dim
        k, v = self.split_heads(_k,mask), self.split_heads(_v,mask)  # [n, h, step, h_dim]
        return q,k,v
    def scaled_dot_product_attention(self, q, k, v,compute_mask=None,mask=None,inputs=None,LM_mask=None):
        
        o1=self.local_attention( q, k[:,:1,:,:], v[:,:1,:,:])
        o2=self.global_attention( q, k, v,compute_mask,mask,inputs,LM_mask)
        return keras.activations.sigmoid(o1)*o2