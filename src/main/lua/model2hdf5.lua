package.path = package.path .. ";src/main/lua/?.lua"

require 'torch'
require 'rnn'
require 'optim'
require 'CmdArgs'
require 'NonProjShiftReduce'
require 'NonProjShiftReducePos'
require 'ProjShiftReduce'
require 'ProjShiftReducePos'
require 'ProjShiftReducePos2'
require 'ParseState'
require 'ParserConstants'
require 'hdf5'

cmd = torch.CmdLine()
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
cmd:option('-save_model', '', 'file to save the trained model to')
cmd:option('-load_model', '', 'file to load trained model from')


local params = cmd:parse(arg)

if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local function to_cuda(x) return params.gpuid >= 0 and x:cuda() or x:double() end

---- set up network ----
local function deserialize_net()
--    local feature_net, class_net
    local model = torch.load(params.load_model)
    local feature_net = to_cuda(model.feature_net)
    local class_net = to_cuda(model.class_net)

    -- criterion
    print(feature_net); print(class_net)
    to_cuda(feature_net); to_cuda(class_net)

    return feature_net, class_net
end

-- currently this only works for adding embeddings (vs concatenating)
local function serialize_model(feature_net, class_net)

    -- nn.LookupTable [torch.DoubleTensor of size 34556x50]
    local word_embeddings = feature_net.modules[1].modules[1].modules[1].weight

    -- nn.Linear(900 -> 200) [torch.DoubleTensor of size 200x900]
    local word_hidden = feature_net.modules[1].modules[1].modules[4].weight:transpose(1,2)
    local word_bias = feature_net.modules[1].modules[1].modules[4].bias

    -- nn.LookupTable [torch.DoubleTensor of size 47x50]
    local pos_embeddings = feature_net.modules[1].modules[2].modules[1].weight

    -- nn.Linear(900 -> 200) [torch.DoubleTensor of size 200x900]
    local pos_hidden = feature_net.modules[1].modules[2].modules[4].weight:transpose(1,2)
    local pos_bias = feature_net.modules[1].modules[2].modules[4].bias

    -- nn.LookupTable [torch.DoubleTensor of size 45x50]
    local label_embeddings = feature_net.modules[1].modules[3].modules[1].weight

    -- nn.Linear(600 -> 200) [torch.DoubleTensor of size 200x600]
    local label_hidden = feature_net.modules[1].modules[3].modules[4].weight:transpose(1,2)
    local label_bias = feature_net.modules[1].modules[3].modules[4].bias

    -- nn.Linear(200 -> 80) [torch.DoubleTensor of size 80x200]
    local output_layer = class_net.modules[1].weight:transpose(1,2)
    local output_bias = class_net.modules[1].bias

    local hd5_output = hdf5.open(params.save_model, 'w')

    hd5_output:write("word_embeddings", word_embeddings)
    hd5_output:write("label_embeddings", label_embeddings)
    hd5_output:write("pos_embeddings", pos_embeddings)
    hd5_output:write("word_hidden", word_hidden)
    hd5_output:write("pos_hidden", pos_hidden)
    hd5_output:write("label_hidden", label_hidden)
    hd5_output:write("word_bias", word_bias)
    hd5_output:write("pos_bias", pos_bias)
    hd5_output:write("label_bias", label_bias)
    hd5_output:write("output_layer", output_layer)
    hd5_output:write("output_bias", output_bias)

    hd5_output:close()
end

local feature_net, class_net = deserialize_net()

serialize_model(feature_net, class_net)
