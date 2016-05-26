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
command="$NNDEPPARSE_ROOT/bin/get-parse-decisions.sh $train_file $dev_file $test_file $decisions_output_dir --feats=$feats --parser=$parser --pos=$pos --lowercase=$lowercase --shape-feats=$shape --suffix-feats=$suffix --stacked=$stacked --embeddings=$embedding_file --maps-dir=$intmaps_output_dir" && \
echo $command && \
eval $command && \
echo "" && \

if [ $collapsed -gt 1 ]; then
    echo "Collapsing decisions (max len = $collapsed)" && \
    command="$NNDEPPARSE_ROOT/bin/collapse-sequential-decisions.sh $train_file $dev_file $test_file $decisions_output_dir $intmaps_output_dir --max-seq-len $collapsed --sliding $sliding --cutoff $cutoff" && \
    echo $command && \
    eval $command && \
    echo ""
fi && \

# Convert treebank feats to ints, create intmaps
echo "Converting decision features to ints..." && \
command="$NNDEPPARSE_ROOT/bin/convert-feats-to-ints.sh $train_file $dev_file $test_file $decisions_output_dir $intmaps_output_dir --feats=$feats" && \
echo $command && \
eval $command && \
echo "" && \

# Convert treebank ints to Torch tensors
echo "Converting decision intmaps to Torch tensors..." && \
command="$NNDEPPARSE_ROOT/bin/convert-feats-to-torch.sh $train_file $dev_file $test_file $intmaps_output_dir $torch_output_dir -feats $feats" && \
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

