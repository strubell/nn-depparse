#!/bin/bash

config=$1
source $config
source $NNDEPPARSE_ROOT/config/global.conf
additional_args=${@:2}

gpuid=`eval $NNDEPPARSE_ROOT/bin/get-free-gpus.sh | tail -1`

model_file="model-$data_name-$config_name.torch"

th $NNDEPPARSE_ROOT/src/main/lua/nn-depparse.lua \
-train_decisions $torch_decisions_train_file \
-test_decisions $torch_decisions_dev_file \
-test_sentences $torch_sentences_dev_file \
-load_embeddings $torch_embeddings_file \
-decision_map $decision_map \
-punct_set $punct_set \
-pos_map $pos2int \
-label_map $label2int \
-word_map $word2int \
-save_model $model_file \
-l2 $l2 \
-lr $lr \
-embedding_dropout $e_dropout \
-hidden_dropout $h_dropout \
-epsilon $epsilon \
-batch_size $batchsize \
-num_epochs $iterations \
-gpuid $gpuid \
-pos_dim $pos_dim \
-word_dim $embedding_dim \
$additional_args

