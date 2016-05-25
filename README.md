nn-depparse
===========
A neural network dependency parser implemented in Torch.

Requirements
------
- sbt 1.13.9
- torch
- torchx
- nn
- optim

Setup
------
1. Set up environment variables. For example, from the root directory of this project on blake:
  ```
    export NNDEPPARSE_ROOT=`pwd`
    export DATA_DIR=/iesl/canvas/strubell/data/
  ```

2. Compile: `sbt compile`
3. Perform all data preprocessing for a given configuration [also compiles]. For example, generate data for a non-projective shift-reduce parser on PTB (configuration file `config/npsr-ptb.conf`): 

  ```
    ./bin/all-data-processing.sh config/npsr-ptb.conf
  ```
4. Tune hyperparameters (assumes a GPU machine and uses all of its GPUs): 

  ```
    ./bin/tune-hyperparams.sh config/npsr-ptb.conf
  ```

Generating training data
-------------
1. Generate parse decisions + features for training from PTB: `./bin/get-parse-decisions-ptb.sh`
2. Generate intmaps from parse decisions + features: `./bin/convert-ptb-feats-to-ints.sh`
3. Generate Torch tensors from intmaps: `./bin/convert-ptb-feats-to-torch.sh`
4. If word intmaps changed, generate Torch embedding tensors: `./bin/convert-collobert-embeddings-to-torch.sh`

Generating test data
-----------
1. Generate dev/test intmaps for each sentence in PTB: `./bin/convert-ptb-sents-to-ints.sh`
2. Generate Torch tensors from sentence intmaps: `./bin/convert-ptb-sents-to-torch.sh`

Luajit hack to allow array-of-size def
-----
In `torch-distro/exe/luajit-rocks/luajit-2.1/src/luajit.c`, add the function:

    static int new_sized_table( lua_State *L )
    {
        int asize = lua_tointeger( L, 1 );
        int hsize = lua_tointeger( L, 2 );
        lua_createtable( L, asize, hsize );
        return( 1 );
    }

in `main`, after `L` is initialized add the lines:

    lua_pushcfunction( L, new_sized_table );
    lua_setglobal( L, "sized_table" );

Reinstall Torch.


