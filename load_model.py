# -*- coding: utf-8 -*-
import os

os.environ["TF_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sznlp.backend import keras, tf

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from pathlib import Path

from bert4keras.tokenizers import Tokenizer

from sznlp.models import Transformer, mask_share_t5_gate
from sznlp.tools import seq2seq_Generate


def list_models():
    cwd = Path.cwd()
    for model in cwd.rglob("*.h5"):
        yield model.relative_to(cwd).as_posix()

def get_writer_model(model_path, support_english: bool = False):
    # 别动，动一下跑不了后果自负
    block_num = 8
    n_head = 8
    maxlen = 500
    argument = {
        "n_head": n_head,
        "model_dim": 64 * n_head * 4,
        "head_dim": 64,
        "max_len": maxlen,
        "drop_rate": 0.1,
        "activation": "relu",
        "output_dim": 64 * n_head,
        "attention_scale": True,
        "center": False,
        "use_bias": False,
        "embeddings_initializer": keras.initializers.TruncatedNormal(stddev=2e-5),
    }

    tokenizer = Tokenizer("vocab.txt", do_lower_case=True)
    model = Transformer(
        encoder_num=block_num,
        decoder_num=block_num,
        encoder_vocab_size=tokenizer._vocab_size + 1,
        encoder_attention="gate_attention_tiny",
        encoder_FFN="FFN_gate",
        encoder_mask_generate=mask_share_t5_gate(
            mask_future=False,
            num_buckets=32,
            max_len=maxlen,
            output_dim=n_head,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        ),
        encoder_mask_future=True,
        decoder_attention="gate_attention_tiny",
        decoder_FFN="FFN_gate",
        decoder_mask_generate=mask_share_t5_gate(
            mask_future=True,
            num_buckets=32,
            max_len=maxlen,
            output_dim=n_head,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        ),
        decoder_mask_future=True,
        output_dims=tokenizer._vocab_size + 1,
        **argument
    ).model(split_model=True)

    # model.summary()
    model.load_weights(model_path)

    encoder = keras.Model(model.inputs[0], model.get_layer("masking_8").output)
    encoder_output = keras.layers.Input(tensor=encoder.output)
    encoder_output = keras.layers.Input(tensor=encoder.output)
    decoder = keras.Model([encoder_output, model.inputs[1]], model.output)
    return seq2seq_Generate(encoder, decoder, tokenizer, start_token=5 if support_english else 4)
