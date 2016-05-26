#!/bin/bash

train_file=$1
dev_file=$2
test_file=$3
data_dir=$4
output_dir=$5
other_args=${@:6}

train_file="$data_dir/${train_file##*/}.decisions.intmap"
test_file="$data_dir/${test_file##*/}.decisions.intmap"
dev_file="$data_dir/${dev_file##*/}.decisions.intmap"

# make directory to contain decisions data if it doesn't exist
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

fnames=( $train_file $test_file $dev_file )
for input_fname in ${fnames[@]}; do
  th $NNDEPPARSE_ROOT/src/main/lua/feats2torch.lua \
  -inFile $input_fname \
  -outFile "$output_dir/${input_fname##*/}.torch" \
  $other_args
done
