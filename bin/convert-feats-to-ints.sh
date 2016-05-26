#!/bin/bash

train_file=$1
dev_file=$2
test_file=$3
data_dir=$4
output_dir=$5
other_args=${@:6}

train_file="$data_dir/${train_file##*/}.decisions"
test_file="$data_dir/${test_file##*/}.decisions"
dev_file="$data_dir/${dev_file##*/}.decisions"

memory=4g

# make directory to contain decisions data if it doesn't exist
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
elif [ -e $decision_map ]; then
  rm $decision_map
fi

# compute and write int maps for training data
$NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx$memory edu.umass.cs.iesl.nndepparse.MapFeatsToInts \
--data-file-file=$train_data_file_file \
--data-file=$train_file \
--output-dir=$output_dir \
--data-output-file="${train_file##*/}.intmap" \
--load-maps=false \
$maps \
$other_args

# for dev and test, want to use int maps we already generated using train data
fnames=( $test_file $dev_file )
for input_fname in ${fnames[@]}; do
  $NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx$memory edu.umass.cs.iesl.nndepparse.MapFeatsToInts \
  --data-file=$input_fname \
  --output-dir=$output_dir \
  --maps-dir=$output_dir \
  --data-output-file="${input_fname##*/}.intmap" \
  --load-maps=true \
  $other_args
done
