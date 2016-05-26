#!/bin/bash

config=$1
source $config
source $NNDEPPARSE_ROOT/config/global.conf
additional_args=${@:2}

h5model_file="model-$data_name-$config_name.h5"

$NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx4g edu.umass.cs.iesl.nndepparse.Parse --data-file=$dev_file --model=$h5model_file --maps-dir=$intmaps_output_dir $additional_args
