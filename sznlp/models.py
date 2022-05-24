# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:42:05 2021
self.args
@author: Administrator
"""
from sznlp.backend import keras,tf,K
from inspect import getfullargspec
from sznlp.layers import *
import threading
from sznlp.layers import relative_position_dict,position_embeiding_dict,attention_dict,FFN_dict,LM_dict
def mask_martic(k):
    #获取常规的mask矩阵
    seq_len=K.shape(k)[-2]
    idxs = K.arange(0, seq_len)
    mask = idxs[None, :] <= idxs[:, None]
    mask = K.cast(mask, K.floatx())
    return 1-mask
class mask_share_t5(RelativePositionEmbeddingT5):
    def __init__(self,mask_future,mask_martic=mask_martic,**kwargs):
        super(mask_share_t5, self).__init__(**kwargs)  
        self.mask_future=mask_future
        self.mask_martic=mask_martic
    def get_config(self):
        config = {
            'mask_future':self.mask_future,
            'mask_martic':self.mask_martic,
            
        }
        base_config = super(mask_share_t5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @recompute_grad
    def call(self, x,mask=None):
        
        length=K.shape(x)[-2]
        if length<self.max_len*2:
            pos_ids =self.index[:length,:length]
        else:
            pos_ids = self.compute_position_ids(length,length)
        position_bias=K.gather(self.embeddings, pos_ids)
        position_bias = K.permute_dimensions(position_bias, (2, 0, 1))
        if self.mask_future:
            LM_mask = -1e9*self.mask_martic(x)
            position_bias=position_bias+LM_mask
        if mask!=None:
            v_mask=K.cast(mask,position_bias.dtype)
            v_mask=1-K.reshape(v_mask,[-1,1,1,K.shape(v_mask)[-1]])
            position_bias+=v_mask*-1e9
        return position_bias#K.exp(position_bias)
class mask_share_t5_gate(mask_share_t5):   
    @recompute_grad
    def call(self, x,mask=None):
        position_bias=super(mask_share_t5_gate, self).call(x,mask=mask)
        return K.exp(position_bias)
class Masking(keras.layers.Layer):
    def call(self, x,mask=None):
        if mask==None:
            return x
        mask=tf.expand_dims(mask,-1)
        mask=tf.cast(mask,x.dtype)
        
        return x*mask
def Resnet(alpha=1):
    if alpha==1:
        class Resnet(keras.layers.Add):
            pass
        return  Resnet()
    class Resnet(keras.layers.Layer):
        def __init__(self,alpha=1,**kwargs):
            self.alpha=alpha
            super(Resnet, self).__init__(**kwargs)  
        def get_config(self):
            config = {
                'alpha':self.alpha,
                
            }
            base_config = super(Resnet, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        def call(self, inputs,mask=None):
            x1,x2=inputs
            return x1+self.alpha*x2
    return Resnet(alpha)
class EncoderBlock():
    def __init__(self,
                 relative_position=None,#所使用的相对位置编码
                 attention='multiattention',#所使用的注意力
                 FFN='FFN',#所使用的FFN
                 LM='LayerNormalization',#所使用的LM
                 pre_norm=False,#post-norm还是pre-norm
                 suffix='',#取名的后缀
                 prefix='encoder_',#取名的前缀
                 alpha=1,
                 **kwargs 
                 ):
        
        self.alpha=alpha
        self.relative_position=relative_position
        self.attention=attention
        self.FFN=FFN
        self.LM=LM
        self.pre_norm=pre_norm
        self.config=kwargs
        self.suffix=suffix
        self.lmnums=0
        self.prefix=prefix
        if 'drop_rate' not in self.config:
            self.config['drop_rate']=0
        self.drop=keras.layers.Dropout(self.config['drop_rate'])
    def get_argument(self,layers):
        #获取不同类的参数
        argument={}
        agrs=[]
        defaults=[]
        while layers!=object:
            configs=getfullargspec(layers.__init__)
            layers=layers.__base__
            arg=configs.args[1:]
            default=configs.defaults
            if len(arg)==0:
                continue
            if len(arg)!=len(default):
                arg=arg[-1*len(default):]
            agrs.extend(arg)
            defaults.extend(default)
        for i in range(1,len(agrs)+1):
            argument[agrs[-i]]=defaults[-i]
        for key in self.config.keys():
            if key in argument.keys():
                argument[key]=self.config[key]
        return argument
    def build_layers(self,layers,name,function=None,prefix=''):
        #实例化层
        if function!=None:
            layers=function(layers)
            name=function.__name__
        argument=self.get_argument(layers)
        argument['name']=self.prefix+prefix+name+self.suffix

        return layers(**argument)
    def layers_call(self,inputs,layers_name,layers_dict,function=None,prefix='',other_inputs=[]):
        #调用层
        if layers_name==None or layers_name=='':
            return inputs
        if type(layers_name)==str:
            layers=layers_dict[layers_name]
        else:
            layers=layers_name
            layers_name=layers.__name__
        if layers_dict==LM_dict:
            layers_name+='_'+str(self.lmnums)
            self.lmnums+=1
        
        layers=self.build_layers(layers, layers_name,function,prefix)
        
        if len(other_inputs)==0:
            output=layers(inputs)
        else:
            
            output=layers(inputs+other_inputs)
            
        return  output
    def apply_attention(self,x,other_inputs=[]):
        #使用attention
        #先判断有无相对位置编码
        if type(self.relative_position)==str:
            relative_fucntion=relative_position_dict[self.relative_position]
        elif self.relative_position!=None:
            relative_fucntion=self.relative_position
        else:
            relative_fucntion=None
        
        outputs=self.layers_call([x,x,x],self.attention,attention_dict,relative_fucntion,prefix='',other_inputs=other_inputs)
        return outputs
    def apply_FFN(self,x,**kwargs):
        outputs=self.layers_call(x,self.FFN,FFN_dict)
        outputs=self.drop(outputs)
        return outputs
    def apply_pre_norm(self,inputs,function,**kwargs):
        inputs=self.layers_call(inputs,self.LM,LM_dict)
        output=function(inputs,**kwargs)
        output=Resnet()([output,inputs])
        return output
    def apply_post_norm(self,inputs,function,**kwargs):
        output=function(inputs,**kwargs)
        output=Resnet(self.alpha)([output,inputs])
        output=self.layers_call(output,self.LM,LM_dict)
        return output
    def apply_all_layers(self,inputs,other_inputs=''):
        #计算的主函数
        if self.pre_norm:
            norm=self.apply_pre_norm
        else:
            norm=self.apply_post_norm
        
        output=norm(inputs,self.apply_attention,other_inputs=other_inputs)
        output=norm(output,function=self.apply_FFN)
        output=Masking()(output)
        return output
class DecoderBlock(EncoderBlock):    
    def __init__(self,cross_attention='multiattention',**kwargs):
        super(DecoderBlock,self).__init__(**kwargs)
        if self.prefix=='encoder_':
            self.prefix='decoder_'
        self.cross_attention=cross_attention
    def apply_cross_attention(self,inputs,encoder_inputs):
        #使用attention
        #先判断有无相对位置编码
        relative_fucntion=None     
        outputs=self.layers_call([inputs,encoder_inputs,encoder_inputs],
                                 self.cross_attention,attention_dict,
                                 relative_fucntion,
                                 prefix='cross_')
        return outputs
    def apply_all_layers(self,inputs,encoder_inputs,other_inputs=[]):
        #计算的主函数
        if self.pre_norm:
            norm=self.apply_pre_norm
        else:
            norm=self.apply_post_norm
        output=norm(inputs,self.apply_attention,other_inputs=other_inputs)
        self.config['mask_future']=False
        output=norm(output,self.apply_cross_attention,encoder_inputs=encoder_inputs)
        output=norm(output,function=self.apply_FFN)
        output=Masking()(output)
        return output
    
class Transformer(EncoderBlock):
    def __init__(self,
                 encoder_num=3,
                 decoder_num=0,
                 #encoderblock和decoderblock的参数
                 encoder_relative_position=None,
                 encoder_attention='multiattention',
                 encoder_FFN='FFN',
                 encoder_LM='LayerNormalization',
                 encoder_pre_norm=False,
                 encoder_prefix='encoder_',
                 encoder_position_embeiding=None,#使用的位置嵌入，默认不存在
                 decoder_position_embeiding=None,#使用的位置嵌入，默认不存在
                 decoder_relative_position=None,
                 decoder_attention='multiattention',
                 decoder_FFN='FFN',
                 decoder_LM='LayerNormalization',
                 decoder_pre_norm=False,
                 decoder_prefix='decoder_',
                 encoder_mask_future=False,
                 decoder_mask_future=True,
                 EncoderBlock=EncoderBlock,
                 DecoderBlock=DecoderBlock,
                 cross_attention='multiattention',
                 #这里是一些其他组件的参数
                 encoder_vocab_size=0,#词典，大于零代表着会使用embeding层
                 decoder_vocab_size=0,
                 #deepnorm config
                 encoder_alpha=1,
                 decoder_alpha=1,
                 segment_size=0,
                 
                 output_dims=None,#最后模型输出的
                 output_activation=None,#最后输出的激励函数
                 word_embeding_size=None,#如果这个参数和中间层不一样，会通过一个dense转换
                 #embeding层的参数
                 pool_layer=None,#一个实例化的pool层
                 finnal_laer=None,#在主体计算完后的最后一层，如CRF。要求是实例化的
                 prefix='',
                 suffix='',
                 encoder_mask_generate=None,
                 decoder_mask_generate=mask_martic,
                 **kwargs 
                 ):
        self.segment_size=segment_size
        self.encoder_mask_generate=encoder_mask_generate
        self.decoder_mask_generate=decoder_mask_generate
        self.pool_layer=pool_layer
        self.finnal_laer=finnal_laer
        #encoder block的参数
        self.encoder_relative_position=encoder_relative_position
        self.encoder_attention=encoder_attention
        self.encoder_FFN=encoder_FFN
        self.encoder_LM=encoder_LM
        self.encoder_pre_norm=encoder_pre_norm
        self.encoder_mask_future=encoder_mask_future
        self.encoder_prefix=encoder_prefix
        self.encoder_num=encoder_num
        self.EncoderBlock=EncoderBlock

        #decoder block的参数
        self.cross_attention=cross_attention
        self.decoder_relative_position=decoder_relative_position
        self.decoder_attention=decoder_attention
        self.decoder_FFN=decoder_FFN
        self.decoder_LM=decoder_LM
        self.decoder_pre_norm=decoder_pre_norm
        self.decoder_mask_future=decoder_mask_future
        self.decoder_prefix=decoder_prefix
        self.decoder_num=decoder_num
        self.DecoderBlock=DecoderBlock
        #层的参数
        self.config=kwargs
        self.encoder_vocab_size=encoder_vocab_size
        self.decoder_vocab_size=decoder_vocab_size
        self.encoder_position_embeiding=encoder_position_embeiding
        self.decoder_position_embeiding=decoder_position_embeiding
        self.output_dims=output_dims
        self.output_activation=output_activation
        self.word_embeding_size=word_embeding_size
        self.prefix=prefix
        self.suffix=suffix
        self.lmnums=0
        self.hidden=self.config['n_head']*self.config['head_dim']
        self.embeding_layers=None
        
        self.encoder_alpha=encoder_alpha,
        self.decoder_alpha=decoder_alpha,
        if 'drop_rate' not in self.config:
            self.config['drop_rate']=0
        self.drop=keras.layers.Dropout(self.config['drop_rate'])
    def get_word_embeding_layer(self,vocab_size,embeding_size,prefix,mask_zero=True):
        name=prefix+'_embdeing'
        layers=keras.layers.Embedding
        argument=self.get_argument(layers)
        argument['name']=name
        argument['mask_zero']=mask_zero
        argument['input_dim']=vocab_size
        argument['output_dim']=embeding_size
        if 'embeddings_initializer' not in self.config.keys():
            argument['embeddings_initializer']=keras.initializers.TruncatedNormal(stddev=0.02)
        
        return layers(**argument)
    def get_dense(self,units,name=None,**kwargs):
        argument=self.get_argument(keras.layers.Dense)
        if 'activation' not in self.config.keys():
            argument['activation']='relu'
        if 'kernel_initializer' not in self.config.keys():
            argument['kernel_initializer']=keras.initializers.TruncatedNormal(stddev=0.02)
        argument['units']=units
        argument['name']=name
        for key in kwargs.keys():
            argument[key]=kwargs[key]
        
        return keras.layers.Dense(**argument)
    def apply_start_layers(self,inputs,vocab_size,embeding_size,prefix,LM,pre_norm,position_embeiding):
        if vocab_size==0 and self.embeding_layers!=None:
            output=self.embeding_layers(inputs)
        elif vocab_size>0:
            if embeding_size==None:
                embeding_size=self.hidden
            self.embeding_layers=self.get_word_embeding_layer(vocab_size, embeding_size, prefix)
            
            output=self.embeding_layers(inputs)
        
        else:
            output=inputs
        if position_embeiding:
            output=self.layers_call(output,position_embeiding,position_embeiding_dict,prefix='')
            #if position_embeiding!='Sinusoidal' and pre_norm==False: 
                #output=self.layers_call(output,LM,LM_dict)
            
        output=self.drop(output)
        if embeding_size!=self.hidden:
               output=self.get_dense(self.hidden)(output)
        output=Masking()(output)
        return output
    
    def apply_final_layers(self,inputs):
        if self.pool_layer!=None:
            inputs=self.pool_layer(inputs)
        inputs=self.get_dense(self.hidden, 'switch_layer')(inputs)
        dense_layer=self.get_dense(self.output_dims, 'output_layer',activation=self.output_activation)
        output=dense_layer(inputs)
        if self.finnal_laer!=None:
            output=self.finnal_laer(output)
        return output

    def get_encoder(self,encoder_inputs=None):
        if encoder_inputs==None:
            encoder_inputs=keras.layers.Input([None],name='encoder_input')
            flag=True
        else:
            flag=False
        
        encoder_output=self.apply_start_layers(encoder_inputs,self.encoder_vocab_size,
                                                 self.word_embeding_size,'encoder',self.encoder_LM,
                                                 self.encoder_pre_norm,self.encoder_position_embeiding)
        if self.segment_size>0:
            segment=keras.layers.Input([None],name='segment_input')
            segment_output=self.get_word_embeding_layer(self.segment_size,self.hidden,'segment_')(segment)
            encoder_output+=segment_output
            encoder_inputs=[encoder_inputs,segment_output]
        other_inputs=[]
        if self.encoder_mask_future:
            mask=self.encoder_mask_generate(encoder_output)  
            other_inputs+=[mask]
        
        for i in range(self.encoder_num):
            suffix='__'+str(i)
            
            argument=self.config
            argument['mask_future']=self.encoder_mask_future

            encoder_output=self.EncoderBlock(                 
                 relative_position=self.encoder_relative_position,
                 attention=self.encoder_attention,
                 FFN=self.encoder_FFN,
                 LM=self.encoder_LM,
                 pre_norm=self.encoder_pre_norm,
                 suffix=suffix,
                 prefix=self.encoder_prefix,
                 alpha=self.encoder_alpha,
                 **argument
                 ).apply_all_layers(encoder_output,other_inputs=other_inputs)
        if flag:
            encoder=keras.models.Model(encoder_inputs,encoder_output,name='encoder_model')
            return encoder
        else:
            return encoder_output
    def get_decoder(self,decoder_inputs=None,encoder_output=None):
        
        if decoder_inputs==None and encoder_output==None:
            flag=True
        else:
            flag=False
        if decoder_inputs==None:
            
            decoder_inputs=keras.layers.Input([None],name='decoder_input')
        if encoder_output==None:
            encoder_output=keras.layers.Input([None,self.hidden],name='encoder_output')
            
        
        decoder_output=self.apply_start_layers(decoder_inputs,self.decoder_vocab_size,
                                              self.word_embeding_size,'decoder',self.decoder_LM,
                                              self.decoder_pre_norm,self.decoder_position_embeiding)

        other_inputs=[]
        if self.decoder_mask_future:
            mask=self.decoder_mask_generate(decoder_output)  
            other_inputs+=[mask]
        
        for i in range(self.decoder_num):
            suffix='__'+str(i)
            argument=self.config
            argument['mask_future']=self.decoder_mask_future
            decoder_output=self.DecoderBlock(                 
                 relative_position=self.decoder_relative_position,
                 attention=self.decoder_attention,
                 FFN=self.decoder_FFN,
                 LM=self.decoder_LM,
                 pre_norm=self.decoder_pre_norm,
                 suffix=suffix,
                 prefix=self.decoder_prefix,
                 cross_attention=self.cross_attention,
                 alpha=self.decoder_alpha,
                 **argument
                 ).apply_all_layers(decoder_output,encoder_output,other_inputs=other_inputs)
        if flag:
            deocder=keras.models.Model([encoder_output,decoder_inputs],decoder_output,name='decoder')
            return deocder
        else:
            return [decoder_inputs,decoder_output]
    def apply_layers(self,encoder_inputs=None,decoder_inputs=None,split_model=False):
        
        #encoder 部分
        if  self.encoder_num>0:
            encoder=self.get_encoder(encoder_inputs)
            output=None
        #decoder部分
        if self.decoder_num>0:
            if split_model:
                decoder_input,decoder_output=self.get_decoder(decoder_inputs,encoder.output)
                output=self.apply_final_layers(decoder_output)
                return output,encoder,decoder_input
            decoder=self.get_decoder(decoder_inputs)
            output=decoder.output
            
        elif encoder_inputs!=None:
            return None,encoder,None
        else:
            output=encoder.output
            decoder=None
        if decoder_inputs!=None:
            return output,encoder,decoder
        if self.output_dims:
            output=self.apply_final_layers(output)
        return output,encoder,decoder
    def model(self,encoder_inputs=None,
              decoder_inputs=None,
              return_model=True,
              name='Transformer',
              split_model=True,
              ):
        output,encoder,decoder=self.apply_layers(encoder_inputs,decoder_inputs,split_model)
        if split_model and self.decoder_num>0:
            if return_model:
                return keras.Model(encoder.inputs+[decoder],output)
            else:
                return output,encoder,decoder
        if encoder_inputs!=None or decoder_inputs!=None:
            return output,encoder,decoder
        if return_model:
            if self.encoder_num>0:
                if  self.decoder_num>0:
                    decoder=keras.models.Model(decoder.inputs,output,name='decoder_model')
                    encoder_output=encoder(encoder.inputs)
                    outputs=decoder([encoder_output,decoder.inputs[1]])
                    model=keras.models.Model(encoder.inputs+[decoder.inputs[1]],outputs,)
                    return model
                else:
                    model=keras.models.Model(encoder.inputs,output)
                    return model
            else:
                outputs=decoder([encoder_inputs,decoder.inputs[1]])
                model=keras.models.Model([encoder.inputs,decoder.inputs[1]],outputs,name=name)
                return model
        else:
            return output,encoder,decoder