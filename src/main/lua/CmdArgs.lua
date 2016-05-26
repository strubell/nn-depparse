
require 'torch'

local CmdArgs = torch.class('CmdArgs')

local cmd = torch.CmdLine()
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
-- data file locations
cmd:option('-train_decisions', 'data/ptb/torch/npsr/wsj02-21-trn.sdep.morph.pos.decisions.intmap.torch','torch format train file list')
cmd:option('-train_sentences', 'data/ptb/torch/npsr/wsj02-21-trn.sdep.morph.pos.sentences.intmap.torch','torch format train file list')
cmd:option('-test_sentences', 'data/ptb/torch/npsr/wsj22-dev.sdep.morph.pos.sentences.intmap.torch', 'torch format test file list')
cmd:option('-test_decisions', 'data/ptb/torch/npsr/wsj22-dev.sdep.morph.pos.decisions.intmap.torch', 'torch format test file list')
cmd:option('-save_model', '', 'file to save the trained model to')
cmd:option('-load_model', '', 'file to load trained model from')
cmd:option('-decision_map', 'data/ptb/intmaps/npsr/int2decision','file containing map from labels to decisions (int -> int int int')
cmd:option('-pos_map', 'data/ptb/intmaps/npsr/pos2int','file containing map from pos strings to ints')
cmd:option('-word_map', 'data/ptb/intmaps/npsr/word2int','file containing map from label strings to ints')
cmd:option('-label_map', 'data/ptb/intmaps/npsr/label2int','file containing map from word strings to ints')

cmd:option('-transition2str', 'data/ptb/intmaps/transition2string','file containing map from word strings to ints')

cmd:option('-punct_set', 'data/ptb/intmaps/npsr/punct','file containing list of punctuation (intmapped pos tags) to skip during eval')
cmd:option('-train_portion', 1.0,'portion of training data to use')
cmd:option('-test_portion', 1.0,'portion of test data to use')

cmd:option('-lua_eval', false, 'Whether to evaluate in lua or scala (lua is extremely slow)')

-- model / data sizes
cmd:option('-word_dim', 50, 'Number of dimensions for word embeddings')
cmd:option('-pos_dim', 50, 'Number of dimensions for pos tag embeddings')
cmd:option('-label_dim', 50, 'Number of dimensions for parse label embeddings')
cmd:option('-hidden_dim', 200, 'parse net hidden layer dimension')
cmd:option('-hidden_layers', 1, 'Number of hidden layers')
cmd:option('-concat_embeddings', false, 'concat initial embeddings together (weiss), otherwise add them after linear (chen)')

-- optimization
cmd:option('-batch_size', 1024, 'minibatch size')
cmd:option('-lr', 0.001, 'init learning rate')
cmd:option('-epsilon', 1e-8, 'epsilon parameter for adam optimization')
cmd:option('-beta1', 0.9, 'beta1 parameter for adam optimization')
cmd:option('-beta2', 0.999, 'beta2 parameter for adam optimization')
cmd:option('-embedding_dropout', 0.0, 'dropout percentage after embedding layer')
cmd:option('-hidden_dropout', 0.5, 'dropout percentage after final hidden layer')
cmd:option('-l2', 1e-8, 'l2 regularization')
cmd:option('-optim', 'adam', 'specify optim method with class name [sgd, adam, adagrad, etc]')
cmd:option('-stop_early', false, 'stop training early if evaluation F1 goes down')
cmd:option('-num_epochs', 10, 'number of epochs to train for')
cmd:option('-evaluate_frequency', 1, 'Evaluate every [n] epochs')
cmd:option('-evaluate_only', false, 'Do not train, just load a model and evaluate')

cmd:option('-fb', false, 'whether we have fb extensions')

cmd:option('-load_embeddings', 'data/ptb/torch/npsr/collobert-embeddings.txt.torch', 'file containing serialized torch embeddings')


function CmdArgs:parse(cmd_args)
    local params = cmd:parse(cmd_args)
    -- print the params in sorted order
    local param_array = {}
    for arg, val in pairs(params) do table.insert(param_array, arg .. ' : ' .. tostring(val)) end
    table.sort(param_array)
    for _, arg_val in ipairs(param_array) do print(arg_val) end
    return params
end