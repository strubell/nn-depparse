#!/bin/bash

config=$1
if [ ! -z "$1" ]; then
  source $config
  source $NNDEPPARSE_ROOT/config/global.conf
fi
additional_args=${@:2}

h5model_file="model-$data_name-$config_name.torch.hd5"

$NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx4g edu.umass.cs.iesl.nndepparse.FactoriePipelineExample \
--model=$h5model_file \
--maps-dir=$intmaps_output_dir \
$additional_args
#--data-file=$dev_file 
