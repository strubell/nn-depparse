#!/bin/bash

memory=256g
train_file=$1
dev_file=$2
test_file=$3
output_dir=$4
other_args=${@:5}

# make directory to contain decisions data if it doesn't exist
if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi

if [ ! -d "$intmaps_output_dir" ]; then
 mkdir -p $intmaps_output_dir
fi

# need to create stacked using training
$NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx$memory edu.umass.cs.iesl.nndepparse.ProcessPTB \
  --data-file=$train_file \
  --data-file-file=$train_data_file_file \
  --output-dir=$output_dir \
  --output-append=decisions \
  --filter-projective=$filter_projective \
  --parallel=$process_parallel \
  --write-map=true \
  $other_args

# compute and write parse decisions
fnames=( $test_file $dev_file )
for input_fname in ${fnames[@]}; do
  $NNDEPPARSE_ROOT/bin/run_class_sbt.sh -Xmx$memory edu.umass.cs.iesl.nndepparse.ProcessPTB \
  --data-file=$input_fname \
  --data-file-file=false \
  --output-dir=$output_dir \
  --output-append=decisions \
  --filter-projective=false \
  $other_args
done
