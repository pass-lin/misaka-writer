# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:22:37 2022

@author: Administrator
"""
model_path='综合.h5'#模型路径
nums=3#开头生成多个下文
#开头
text='''
却说张飞带着几人与那鲁智深厮杀起来，那鲁智深使了个开碑裂石的招数，把那张飞一把丢在地上，自己落了地，往外便跑。
　到了门口，被他的随从一个劲的追杀。张飞心中一怒，便一刀砍在一名将军的腿上，那名将军疼的大叫着往前一滚，张飞一个转身，一脚便踢在他的胸口上，那人一下子滚到了台阶边，也是摔了个四脚朝天。
　张飞不敢恋战，赶紧带了几个人从后门离去，一路上还不断的大叫：
　“鲁智深，我与你不共戴天！”

'''
import json
import os
os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sznlp.backend import set_gelu,tf,keras
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from sznlp.models import Transformer,mask_share_t5_gate
from bert4keras.tokenizers import Tokenizer
from sznlp.tools import seq2seq_Generate

def get_writer_model():
    #别动，动一下跑不了后果自负
    block_num=8
    n_head=8
    maxlen=500
    argument={'n_head':n_head,
              'model_dim':64*n_head*4,
              'head_dim':64,
              'max_len':maxlen,
              'drop_rate':0.1,
              'activation':'relu',
               'output_dim':64*n_head,
              'attention_scale':True,
              'center':False,
              'use_bias':False,
              'embeddings_initializer':keras.initializers.TruncatedNormal(stddev=2e-5),
              }
    
    
    tokenizer=Tokenizer('vocab.txt', do_lower_case=True)
    model=Transformer(encoder_num=block_num,
                           decoder_num=block_num,
                           encoder_vocab_size=tokenizer._vocab_size+1,
                          encoder_attention='gate_attention_tiny',
                          encoder_FFN='FFN_gate',
                           encoder_mask_generate=mask_share_t5_gate(mask_future=False,num_buckets=32,
                                                  max_len=maxlen,
                                                  output_dim=n_head,
                                                  embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02)),
                           encoder_mask_future=True,
                           
                           decoder_attention='gate_attention_tiny',
                          decoder_FFN='FFN_gate',
                           decoder_mask_generate=mask_share_t5_gate(mask_future=True,num_buckets=32,
                                                  max_len=maxlen,
                                                  output_dim=n_head,
                                                  embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02)),
                           decoder_mask_future=True,
                           output_dims=tokenizer._vocab_size+1,
                           **argument).model(split_model=True)
    
    model.load_weights(model_path)
    #model.summary()
    
    
    encoder=keras.Model(model.inputs[0],model.get_layer('masking_8').output)
    encoder_output=keras.layers.Input(tensor=encoder.output)
    encoder_output=keras.layers.Input(tensor=encoder.output)
    decoder=keras.Model([encoder_output,model.inputs[1]],model.output)
    return seq2seq_Generate(encoder,decoder,tokenizer,start_token=4)


#使用方法
generate= get_writer_model() #这样子获得模型
import time
start=time.time()

#输入，建议开头字数在50字到200字之间


result=generate.writer([text.replace('\n', '氼')],#文本数据就是上面的data
               nums=nums,#一个开头要生成几个文本
               k=0.8,#搜索窗口
               batch_size=32,
               max_len=512,#最大长度
               iter_data_num=400,#一次处理多少个开头
               mode='topp',#别动
               iter_max_num=0,)#检查重复解码的句子的次数，越大就越慢同时重复句子越少)
end=time.time()
s=''
for t in text.split('\n'):
    s+='\t'+t+'\n'
text=s
for i in range(nums):
    print(text)
    for t in result[i].split('氼'):
        print('\t'+t)
    print('*******************************************************************************')
