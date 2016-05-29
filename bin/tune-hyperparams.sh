#!/bin/bash

config=$1
source $config
source $NNDEPPARSE_ROOT/config/global.conf

additional_args=${@:2}

timestamp=`date +%Y-%m-%d-%H-%M-%S`

ROOT_DIR=$NNDEPPARSE_ROOT
OUT_LOG=$ROOT_DIR/hyperparams/$data_name-$config_name-$timestamp

echo "Writing to "$OUT_LOG

mkdir -p $OUT_LOG

iters="10"
concat="" #-concat_embeddings"

args="-train_decisions $torch_decisions_train_file -test_decisions $torch_decisions_dev_file -test_sentences $torch_sentences_dev_file -load_embeddings $torch_embeddings_file -decision_map $decision_map -punct_set $punct_set -pos_map $pos2int -label_map $label2int -word_map $word2int -word_dim $embedding_dim -num_epochs $iters $additional_args"

echo "Using args: $args"

# run on all available gpus
gpuids=( `eval $NNDEPPARSE_ROOT/bin/get-free-gpus.sh | sed '1d'` )
num_gpus=${#gpuids[@]}

# grid search over these
lrs="0.001" # 0.0005 0.01"
h_dropouts="0.0 0.25 0.5"
e_dropouts="0.5 0.0 0.25"
epsilons="1e-8"
l2s="1e-10 1e-8 1e-6 1e-4"
batchsizes="1024"
pos_dims="50"
label_dims="50"

# array to hold all the commands we'll distribute
declare -a commands

# first make all the commands we want
for h_dropout in $h_dropouts; do
   for lr in $lrs; do
       for l2 in $l2s; do
           for batchsize in $batchsizes; do
               for e_dropout in $e_dropouts; do
                    for epsilon in $epsilons; do
                        for pos_dim in $pos_dims; do
                            for label_dim in $label_dims; do
                                fname_append="$lr-$h_dropout-$e_dropout-$l2-$epsilon-$batchsize-p$pos_dim-l$label_dim" \
                                commands+=("th src/main/lua/nn-depparse.lua \
                                $args \
                                -hidden_dropout $h_dropout \
                                -lr $lr \
                                -l2 $l2 \
                                -epsilon $epsilon \
                                -batch_size $batchsize \
                                -embedding_dropout $e_dropout \
                                -label_dim $label_dim \
                                -pos_dim $pos_dim \
				-save_model model-$fname_append.torch \
                                -gpuid XX \
                                &> $OUT_LOG/train-$fname_append.log")
                            echo "Adding job lr=$lr hidden_dropout=$h_dropout embedding_dropout=$e_dropout l2=$l2 batchsize=$batchsize epsilon=$epsilon label_dim=$label_dim pos_dim=$pos_dim"
                                done
                            done
                    done
                done
           done
       done
   done
done

# now distribute them to the gpus
#
# currently this is only correct if the number of jobs is a 
# multiple of the number of gpus (true as long as you have hyperparams
# ranging over 2, 3 and 4 values)!
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for gpuid in ${gpuids[@]}; do
    for (( i=0; i<$jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]/XX/$gpuid}"
        echo "Starting job $jobid on gpu $gpuid"
        eval ${comm}
    done &
    j=$((j + 1))
done
