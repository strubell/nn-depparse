#!/bin/bash

train_file=$1
dev_file=$2
test_file=$3
maps_dir=$4
output_dir=$5
other_args=${@:6}

memory=4g

# make directory to contain decisions data if it doesn't exist
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

# compute and write int maps for training data
$NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx$memory edu.umass.cs.iesl.nndepparse.MapSentsToInts \
--data-file-file=$train_data_file_file \
--data-file=$train_file \
--output-dir=$output_dir \
--data-output-file="${train_file##*/}.sentences.intmap" \
--load-maps=true \
--maps-dir=$maps_dir \
$other_args

# for dev and test, want to use int maps we already generated using train data
fnames=( $test_file $dev_file )
for input_fname in ${fnames[@]}; do
  $NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx$memory edu.umass.cs.iesl.nndepparse.MapSentsToInts \
  --data-file=$input_fname \
  --output-dir=$output_dir \
  --maps-dir=$maps_dir \
  --load-maps=true \
  --data-output-file="${input_fname##*/}.sentences.intmap" \
  $other_args
done
