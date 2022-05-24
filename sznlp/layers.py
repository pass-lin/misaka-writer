# -*- LayerNormalizationcoding: utf-8 -*-
"""
Created on Thu Aug  5 14:40:49 2021

@author: Administrator
"""


from sznlp.Attention import *
from sznlp.Other_layers import *
#当前支持的位置相对位置编码，相对位置编码应该是一个函数，输入是attention，返回是拥有这个相对功能的attention
share_position_dict={
    'R_transformer':LocalRNN_layer,
    'rnn':keras.layers.SimpleRNN
    }
relative_position_dict={'rope':Roformer,
               'roformer':Roformer,
               'rotary':Roformer,
               'deberta':Deberta_attention,
               't5':T5_relative,
               'R_Transformer':R_Transformer,
               'distance': Distance,
               'sumformer':sumformer,
               'R_Transformer':R_Transformer,
               
    }
#当前指出的位置嵌入
position_embeiding_dict={'vanilla':SinusoidalPositionEmbedding,
               'Sinusoidal':SinusoidalPositionEmbedding,
               'embeding':PositionEmbedding,
               'LocalRNN_layer':LocalRNN_layer,

               }
#当前支持的attention及其变种
attention_dict={'attention':MultiHead,
               'multiattention':MultiHead,
               'AFT':AFT_full,
               'AFT_full':AFT_full,
               'Synthesizer_R':Synthesizer_R,
               'Synthesizer':Synthesizer_R,
               'Nystromformer':Nyströmformer,
               'AFT_gate':AFT_gate,
               'gate_attention':gate_attention,
               'AFT_relative':AFT_relative,
               'gate_attention_tiny':gate_attention_tiny,
               'gate_attention_tiny_one_head':gate_attention_tiny_one_head,
               }
FFN_dict={'FFN':PositionWiseFFN,
          'FFN_gate':FFN_gate,
    }
LM_dict={'LM':LayerNormalization,
         'LN':LayerNormalization,
         'LayerNormalization':LayerNormalization,
          
    }