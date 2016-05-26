#!/bin/bash

config=$1
source $config
source $NNDEPPARSE_ROOT/config/global.conf
additional_args=${@:2}

gpuid=`eval $NNDEPPARSE_ROOT/bin/get-free-gpus.sh | tail -1`

model_file="model-$data_name-$config_name.torch"
h5model_file="model-$data_name-$config_name.h5"

th $NNDEPPARSE_ROOT/src/main/lua/model2hdf5.lua -load_model $model_file -gpuid $gpuid -save_model $h5model_file $torch_shape_param -suffix $suffix $additional_args
