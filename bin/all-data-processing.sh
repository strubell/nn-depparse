# TODO this should be a Makefile!

config=$1

source $config
source $NNDEPPARSE_ROOT/config/global.conf

# Re-compile Scala for treebank -> feats processing
echo "Building Scala sources..." && \
command="sbt compile" && \
echo $command && \
eval $command && \
echo "" && \

# Treebank -> feats processing
echo "Converting treebank to decision features..." && \
command="$NNDEPPARSE_ROOT/bin/get-parse-decisions.sh $train_file $dev_file $test_file $decisions_output_dir --pos=$pos --lowercase=$lowercase --maps-dir=$intmaps_output_dir --embeddings=$embedding_file" && \
echo $command && \
eval $command && \
echo "" && \

# Convert treebank feats to ints, create intmaps
echo "Converting decision features to ints..." && \
command="$NNDEPPARSE_ROOT/bin/convert-feats-to-ints.sh $train_file $dev_file $test_file $decisions_output_dir $intmaps_output_dir" && \
echo $command && \
eval $command && \
echo "" && \

# Convert treebank ints to Torch tensors
echo "Converting decision intmaps to Torch tensors..." && \
command="$NNDEPPARSE_ROOT/bin/convert-feats-to-torch.sh $train_file $dev_file $test_file $intmaps_output_dir $torch_output_dir" && \
echo $command && \
eval $command && \
echo "" && \

# Convert treebank sentences to ints [using intmaps]
echo "Converting treebank sentences to ints..." && \
command="$NNDEPPARSE_ROOT/bin/convert-sents-to-ints.sh $train_file $dev_file $test_file $intmaps_output_dir $intmaps_output_dir --lowercase=$lowercase" && \
echo $command && \
eval $command && \
echo "" && \

# Convert treebank sentence ints to Torch
echo "Converting treebank sentence ints to Torch tensors..." && \
command="$NNDEPPARSE_ROOT/bin/convert-sents-to-torch.sh $train_file $dev_file $test_file $intmaps_output_dir $torch_output_dir" && \
echo $command && \
eval $command && \
echo "" && \

# Convert embeddings to Torch tensors
echo "Converting embeddings to Torch tensors..." && \
command="$NNDEPPARSE_ROOT/bin/convert-embeddings-to-torch.sh $embedding_file $intmaps_output_dir $torch_output_dir $embedding_dim" && \
echo $command && \
eval $command && \
echo ""

