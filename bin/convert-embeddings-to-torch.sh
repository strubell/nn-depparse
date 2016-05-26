#!/usr/bin/env bash

embedding_file=$1
vocab_dir=$2
output_dir=$3
dim=$4

output_file="$output_dir/${embedding_file##*/}.torch"
vocab_file="$vocab_dir/word2int"

# make directory to contain decisions data if it doesn't exist
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

th $NNDEPPARSE_ROOT/src/main/lua/WordEmbedding2Torch.lua \
-embeddingFile $embedding_file \
-outFile $output_file \
-vocabFile $vocab_file \
-dim $dim
