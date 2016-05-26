package.path = package.path .. ";src/main/lua/?.lua"

require 'torch'
require 'torchx'
require 'optim'
require 'CmdArgs'
require 'ProjShiftReduce'
require 'ParseState'
require 'ParserConstants'

local params = CmdArgs:parse(arg)
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local function to_cuda(x) return params.gpuid >= 0 and x:cuda() or x end

torch.manualSeed(0)
math.randomseed(0)

local train_decisions = torch.load(params.train_decisions)
local dev_sentences = torch.load(params.test_sentences)
local dev_decisions = torch.load(params.test_decisions)

local word_vocab_size = train_decisions.word_domain_size
local pos_vocab_size = train_decisions.pos_domain_size
local label_vocab_size = math.max(train_decisions.label_domain_size, dev_decisions.label_domain_size)
local words_per_example = train_decisions.num_word_feats
local pos_per_example = train_decisions.num_pos_feats
local labels_per_example = train_decisions.num_label_feats
local decision_domain_size = train_decisions.decision_domain_size

local num_train_decisions = math.floor(#train_decisions * params.train_portion)
print(string.format("Using %d training sentences", num_train_decisions))

local num_dev_sentences = math.floor(#dev_sentences * params.test_portion)
print(string.format("Using %d dev sentences", num_dev_sentences))

local train_decisions_portion = {}
if(num_train_decisions == #train_decisions) then
    train_decisions_portion = train_decisions
else
    for i=1,num_train_decisions do
        table.insert(train_decisions_portion, train_decisions[i])
    end
end

local dev_sentences_portion = {}
local dev_decisions_portion = {}
if(num_dev_sentences == #dev_sentences) then
    dev_sentences_portion = dev_sentences
    dev_decisions_portion = dev_decisions
else
    for i=1,num_dev_sentences do
        table.insert(dev_sentences_portion, dev_sentences[i])
        table.insert(dev_decisions_portion, dev_decisions[i])
    end
end

local num_train_examples = 0
for i = 1,num_train_decisions do
    num_train_examples = num_train_examples + train_decisions[i].decisions:size(1)
end

local num_dev_examples = 0
for i = 1,num_dev_sentences do
    num_dev_examples = num_dev_examples + dev_decisions[i].decisions:size(1)
end

local num_feats = words_per_example + pos_per_example + labels_per_example

local in_dim = (params.word_dim * words_per_example) +
        (params.pos_dim * pos_per_example) +
        (params.label_dim * labels_per_example)

Constants = ParserConstants(params.pos_map, params.label_map, params.word_map)

--- optimization parameters
local opt_config = {
    learningRate = params.lr, epsilon = params.epsilon,
    beta1 = params.beta1, beta2 = params.beta2,
    momentum = params.momentum, learningRateDecay = params.delta
}
local opt_state = {}

local function make_lookup_table(vocab_size, embedding_size, fb)
    local lookup_table
    if(fb) then
        require 'fbcunn'
        lookup_table = nn.LookupTableGPU(vocab_size, embedding_size, false)
    else
        lookup_table = nn.LookupTable(vocab_size, embedding_size)
    end
    -- initialize in range [-.01, .01]
    lookup_table.weight = torch.rand(vocab_size, embedding_size):add(-.01):mul(0.01)
    return lookup_table
end

local function build_embedding_lookup_table(num_embeddings, embedding_size, vocab_size, embeddings_file, dropout_rate, output_dim)
    local embedding_table = make_lookup_table(vocab_size, embedding_size, params.fb)
    if embeddings_file ~= '' then
        print('preloading embeddings from ' .. embeddings_file)
        local data = torch.load(embeddings_file)
        word_vocab_size = data:size(1)
        if(embedding_size ~= data:size(2)) then print(string.format("Loaded embedding size %d does not match given embedding size %d", data:size(2), embedding_size)) end
        embedding_table.weight = data
    end
    local input = nn.Sequential():add(embedding_table):add(nn.Reshape(embedding_size * num_embeddings, true))
    if dropout_rate > 0.0 then input:add(nn.Dropout(dropout_rate)) end
    if not params.concat_embeddings then input:add(nn.Linear(embedding_size * num_embeddings, output_dim)) end
    return input
end

---- set up network ----
local function build_net()
    local net
    if(params.load_model ~= '') then
        local model = torch.load(params.load_model)
        net = model.net
        opt_state = model.opt_state
        for key, val in pairs(opt_state) do if (torch.type(val) == 'torch.DoubleTensor') then opt_state[key] = to_cuda(val) end; end
    else
        -- create the 3 lookup tables for words, pos, and labels
        -- a bunch of gross stuff is added to deal with reshaping and if any are not used
        local inputs = nn.ParallelTable()
        if words_per_example > 0 then
            inputs:add(build_embedding_lookup_table(words_per_example, params.word_dim, word_vocab_size, params.load_embeddings, params.embedding_dropout, params.hidden_dim))
        end
        if pos_per_example > 0 then
            inputs:add(build_embedding_lookup_table(pos_per_example, params.pos_dim, pos_vocab_size, '', params.embedding_dropout, params.hidden_dim))
        end
        if labels_per_example > 0 then
            inputs:add(build_embedding_lookup_table(labels_per_example, params.label_dim, label_vocab_size, '', params.embedding_dropout, params.hidden_dim))
        end

        local feature_net = nn.Sequential()
        feature_net:add(inputs)
        if not params.concat_embeddings then
            feature_net:add(nn.CAddTable())
        else
            feature_net:add(nn.JoinTable(2))
                :add(nn.Reshape(in_dim))
                :add(nn.Linear(in_dim, params.hidden_dim))
        end

        feature_net:add(nn.ReLU())

        for i = 2, params.hidden_layers do
            feature_net:add(nn.Linear(params.hidden_dim, params.hidden_dim))
            if params.bias ~= 0 then feature_net:add(nn.Add(params.hidden_dim, params.bias)) end
            feature_net:add(nn.ReLU())
        end

        local class_net = nn.Sequential():add(nn.Linear(params.hidden_dim, decision_domain_size))
        if params.hidden_dropout > 0.0 then class_net:add(nn.Dropout(params.hidden_dropout)) end
        class_net:add(nn.LogSoftMax())

        net = nn.Sequential():add(feature_net):add(class_net)
    end

    -- criterion
    local criterion = nn.ClassNLLCriterion()
    to_cuda(criterion)

    print(net)
    to_cuda(net)

    return net, criterion
end

--- Evaluate ---
local function predict(example, net, criterion)
    local pred = net:forward(example)
    local max_scores, max_labels = pred:max(2)
    return max_labels
end

local function evaluate(data, net, criterion)
    local token_correct = 0.0
    local token_total = 0.0
    for i=1,#data do
        local example = { to_cuda(data[i].word_feats), to_cuda(data[i].pos_feats), to_cuda(data[i].label_feats) }
        local pred_labels = predict(example, net, criterion)
        pred_labels = to_cuda(pred_labels)
        local gold_labels = to_cuda(data[i].decisions:long())
        local sent_correct = pred_labels:eq(gold_labels):sum()
        token_correct = token_correct + sent_correct
        token_total = token_total + pred_labels:size(1)
    end
    return token_correct/token_total
end

-- returns las, uas aggregated over data
-- punct is a set of pos tags we consider puntuation to skip ( `` '' : , . )
local function evaluate_parse(data, parser, punct)
    local las_correct = 0.0
    local uas_correct = 0.0
    local total = 0.0
    local pos_correct = 0.0
    local pos_total = 0.0
    local pos_confusion = torch.Tensor(pos_vocab_size, pos_vocab_size):zero()
    local label_confusion = torch.Tensor(label_vocab_size, label_vocab_size):zero()
    local decision_confusion = torch.Tensor(decision_domain_size, decision_domain_size):zero()
    for i=1,#data do
        local sentence = data[i]
        local pred_heads, pred_labels = parser:parse(sentence, params.feats)
        pred_labels = to_cuda(pred_labels)
        pred_heads = to_cuda(pred_heads)
        local pred_pos = to_cuda(sentence:select(2,2))
        local gold_labels = to_cuda(sentence:select(2,3))
        local gold_heads = to_cuda(sentence:select(2,4))
        local gold_pos = to_cuda(sentence:select(2,5))
        local non_punct = to_cuda(torch.Tensor(sentence:size(1)):zero())
        for i = 1,sentence:size(1) do
            if(not punct[gold_pos[i]]) then non_punct[i] = 1 end
        end

        -- pos accuracy, confusion
        local pos_correct_elems = to_cuda(pred_pos:eq(gold_pos))
        pos_correct = pos_correct + pos_correct_elems:sum()
        pos_total = pos_total + sentence:size(1)
        for i = 1,sentence:size(1) do
            pos_confusion[gold_pos[i]][pred_pos[i]] = pos_confusion[gold_pos[i]][pred_pos[i]] + 1
        end

        -- label confusion
        for i = 1,sentence:size(1) do
            if(pred_labels[i] ~= 0) then
                label_confusion[gold_labels[i]][pred_labels[i]] = label_confusion[gold_labels[i]][pred_labels[i]] + 1
            end
        end

        local sent_label_correct = to_cuda(pred_labels:eq(gold_labels):double())
        local sent_head_correct = to_cuda(pred_heads:eq(gold_heads):double())

        -- gross hack in order to element-wise AND: sum + eq w/ 2/3 (for punct)
        local sent_head_sum = to_cuda(sent_head_correct:add(non_punct))
        local sent_label_sum = to_cuda(sent_label_correct:add(sent_head_sum))

        local sent_head_correct_nopunct = to_cuda(sent_head_sum:eq(to_cuda(torch.Tensor(sentence:size(1))):fill(2)):double())
        local sent_label_correct_nopunct = to_cuda(sent_label_sum:eq(to_cuda(torch.Tensor(sentence:size(1))):fill(3)):double())

        las_correct = las_correct + sent_label_correct_nopunct:sum()
        uas_correct = uas_correct + sent_head_correct_nopunct:sum()
        total = total + non_punct:sum() -- only count toks not marked as punctuation
    end
    return las_correct/total, uas_correct/total, pos_correct/pos_total, pos_confusion, label_confusion, decision_confusion
end

local function test_feats(feature_net, class_net, sent_data, decision_data, parser, punct)

    -- hack to turn of bias in loaded model
--    feature_net.modules[3].bias:fill(0)

    -- for readable debugging output
    local transition2string = {}
    for line in io.lines(params.transition2str) do
        if(line ~= "") then
            local str, number = string.match(line, "([^\t]+)\t([^ ]+)")
            transition2string[tonumber(number)] = str
        end
    end
    local label2string = {}
    for line in io.lines(params.label_map) do
        if(line ~= "") then
            local str, number = string.match(line, "([^\t]+)\t([^ ]+)")
            label2string[tonumber(number)] = str
        end
    end

    local pos2string = {}
    for line in io.lines(params.pos_map) do
        if(line ~= "") then
            local str, number = string.match(line, "([^\t]+)\t([^ ]+)")
            pos2string[tonumber(number)] = str
        end
    end

--    for i=267,267 do
    for i=1,1 do
        local sentence = sent_data[i]
        local decisions = decision_data[i]
        print("sent_data len: ", #sent_data, "decisions len: ", #decision_data)
        local num_gold_decisions = decisions['decisions']:size(1)
        local state = ParseState(1, 2, {}, sentence)
        local j = 1
        while (state.input <= state.parseSentenceLength or (state.stack > 1 and state.stack <= state.parseSentenceLength)) do
            print(state.stack, state.input, state.parseSentenceLength, j)
            if(state.stack < 1) then
                state.stack = state.input
                state.input = state.input + 1
            else
                -- make feats for prediction here
                local feats
                if(params.feats == "iesl") then
                    feats = parser:compute_features_iesl(state)
                elseif(params.feats == "chen") then
                    feats = parser:compute_features_chen(state)
                elseif(params.feats == "bcon") then
                    feats = parser:compute_features_bcon(state)
                elseif(params.feats == "bcon2") then
                    feats = parser:compute_features_bcon2(state)
                end

                if(#sent_data == #decision_data and j <= num_gold_decisions) then
                    if(joint_parser) then
                        local true_d1, true_d2, true_d3, true_d4 = parser:parse_decision(decisions['decisions'][j], 1)
                        print(string.format("true decision: %s %s %s %s", transition2string[true_d1], transition2string[true_d2], label2string[true_d3], pos2string[true_d4]))
                    else
                        print("decision:", decisions['decisions']:size(1))
                        local true_d1, true_d2, true_d3 = parser:parse_decision(decisions['decisions'][j], 1)
                        print(string.format("true decision: %s %s %s", transition2string[true_d1], transition2string[true_d2], label2string[true_d3]))
                    end

                    print("true word feats:")
                    print(decisions['word_feats'][j])
                    print("pred word feats:")
                    print(feats[1])

                    print("true pos feats:")
                    print(decisions['pos_feats'][j])
                    print("pred pos feats:")
                    print(feats[2])

                    print("true label feats:")
                    print(decisions['label_feats'][j])
                    print("pred label feats:")
                    print(feats[3])

                    if(params.shape) then
                        print("true shape feats:")
                        print(decisions['shape_feats'][j])
                        print("pred shape feats:")
                        print(feats[4])
                    end

                    if(params.suffix > 0) then
                        print("true suffix feats:")
                        print(decisions['suffix_feats'][j])
                        print("pred suffix feats:")
                        print(feats[5])
                    end

                else
--                    print("No gold decision")

                    print("pred word feats:")
                    print(feats[1])

                    print("pred pos feats:")
                    print(feats[2])

                    print("pred label feats:")
                    print(feats[3])

                    if(params.shape) then
                        print("pred shape feats:")
                        print(feats[4])
                    end

                    if(params.sufix > 0) then
                        print("pred suffix feats:")
                        print(feats[5])
                    end
                end

                local pred = parser.net:forward(feats)

--                print("embedded word:")
--                print(feature_net.modules)

--                print("add output:")
--                print(feature_net.modules[3].output)

--                print("word output:")
--                print(feature_net.modules[1].modules[1].modules[4].output)

--                print("word input:")
--                print(feature_net.modules[1].modules[1].modules[1].output)

--                                print("label output:")
--                                print(feature_net.modules[1].modules[3].output)

--                print("word hidden:")
--                 print(feature_net.modules[1].modules[1].modules[4].weight)

--                print("relu:")
--                print(feature_net.modules[4].output)

--                print("scores:")
--                print(class_net.modules[1].output)
                state:print()
                local _, decision = pred:max(2)
                for i=1,params.collapsed do
                    --                local decisionPart = string.match(decision, self.decision_regex)
                    if((state.input <= state.parseSentenceLength or state.stack > 1)) then
                        if(state.stack < 1) then parser:shift(state) end
                        local leftOrRightOrNo, shiftOrReduceOrPass, label = parser:parse_decision(decision[1][1], i)
                        print(string.format("pred decision: %s %s %s", transition2string[leftOrRightOrNo], transition2string[shiftOrReduceOrPass], label2string[label]))
                        if(leftOrRightOrNo ~= 0 and (state.input ~= state.parseSentenceLength+1 or not (shiftOrReduceOrPass == Constants.SHIFT and leftOrRightOrNo == Constants.NO))) then
                            try {
                                function()
                                    --                            print("decision:", leftOrRightOrNo, shiftOrReduceOrPass, label)
                                    parser:transition(state, leftOrRightOrNo, shiftOrReduceOrPass, label)
                                end,
                                catch {
                                    function(error)
                                        print('caught error: ' .. error)
                                        print("decision", leftOrRightOrNo, shiftOrReduceOrPass, label)
                                        state:print()
                                        state.stack = 1; state.input = state.parseSentenceLength+1
                                    end
                                }
                            }
                        else
                            state.stack = 1
                            state.input = state.input + 1
                        end
                    end
                end
--
--
--                state:print()
--                local leftOrRightOrNo, shiftOrReduceOrPass, label, pos = parser:parse_decision(decision[1][1])
--                try {
--                    function()
--                        parser:transition(state, leftOrRightOrNo, shiftOrReduceOrPass, label, pos)
--                    end,
--                    catch {
--                        function(error)
--                            print('caught error: ' .. error)
--                            print("decision", leftOrRightOrNo, shiftOrReduceOrPass, label, pos)
--                            state:print()
--                            state.stack = 1; state.input = state.parseSentenceLength+1
--                        end
--                    }
--                }
            end
            j = j + 1
        end
        local pred_heads = torch.add(state.headIndices:narrow(1, 2, state.parseSentenceLength-1), -2)
        local pred_labels = state.arcLabels:narrow(1, 2, state.parseSentenceLength-1)
        local pred_pos = sentence:select(2, 2)

        print(sentence)

        local gold_labels = sentence:select(2,3)
        local gold_heads = sentence:select(2,4)
        local gold_pos = sentence:select(2, 5)
        local stacked_pos = sentence:select(2, 6)
        local non_punct = to_cuda(torch.Tensor(sentence:size(1)):zero())
        for i = 1,sentence:size(1) do
            if(not punct[gold_pos[i]]) then non_punct[i] = 1 end
        end

        print("pred", "", "gold", "", "a_pos", "g_pos", "s_pos", "punct")
        for j=1,sentence:size(1) do
            print(pred_heads[j], pred_labels[j], gold_heads[j], gold_labels[j], pred_pos[j], gold_pos[j], stacked_pos[j], non_punct[j])
        end
        print("\n")

    end

end

local function get_batch_data(train_decisions)

    local num_clusters = 1

    local decisions = {}
    local word_feats = {}
    local pos_feats= {}
    local label_feats = {}
    local shape_feats = {}
    local suffix_feats = {}

    -- load in all the examples
    for i = 1, #train_decisions do
        table.insert(decisions, train_decisions[i].decisions)
        table.insert(word_feats, train_decisions[i].word_feats)
        table.insert(pos_feats, train_decisions[i].pos_feats)
        table.insert(label_feats, train_decisions[i].label_feats)
    end

    -- flatten all the examples into tensors
    local decisions_tensor = to_cuda(nn.JoinTable(1)(decisions))
    local word_tensor = to_cuda(nn.JoinTable(1)(word_feats))
    local pos_tensor = to_cuda(nn.JoinTable(1)(pos_feats))
    local label_tensor = to_cuda(nn.JoinTable(1)(label_feats))

    return decisions_tensor, word_tensor, pos_tensor, label_tensor
end


--- split training data into batches
local function gen_batches(train_decisions, batch_size)

    local decisions_tensor, word_tensor, pos_tensor, label_tensor = get_batch_data(train_decisions)

    -- divide each cluster into batches
    local batches = {}
    local start = 1
    local num_examples = decisions_tensor:size(1)
    local rand_order = torch.randperm(num_examples):long()
    while(start <= num_examples) do
        local size = math.min(batch_size, num_examples - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local decision_batch = decisions_tensor:index(1, batch_indices)
        local word_batch = word_tensor:index(1, batch_indices)
        local pos_batch = pos_tensor:index(1, batch_indices)
        local label_batch = label_tensor:index(1, batch_indices)
        table.insert(batches, {data = { word_batch, pos_batch, label_batch }, label = decision_batch})
        start = start + size
    end

    return batches
end

local function serialize_model(net, optim_state)
    local cpu_opt_state = {}
    if params.save_model ~= '' then
        for key, val in pairs(optim_state) do
            if (torch.type(val) == 'torch.CudaTensor') then cpu_opt_state[key] = val:clone():double()
            else cpu_opt_state[key] = val end
        end
        net:clearState()
        torch.save(params.save_model, {net = net:clone():double(), opt_state = cpu_opt_state})
    end
end

-- table of intmapped pos tags that correspond to punctuation (a la Chen & Manning 2014)
local function load_punct(fname)
    local punct_table = {}
    for line in io.lines(fname) do
        if(line ~= "") then
            punct_table[tonumber(line)] = true
        end
    end
    return punct_table
end

local function print_evaluation(net, criterion, test_sentences, test_decisions, parser, punct, print_confusion)
    print("Evaluating")
    -- for pretend accuracy
    local accuracy = evaluate(test_decisions, net, criterion)
--    local train_accuracy = evaluate(train, net)
--    print(string.format('Train decision accuracy: %2.2f', train_accuracy*100))
    print(string.format('Test decision accuracy: %2.2f', accuracy*100))

    -- for actual parsing
    local las, uas, pos, pos_confusion, label_confusion = evaluate_parse(test_sentences, parser, punct)
    print(string.format('Test LAS: %2.2f UAS: %2.2f POS: %2.2f', las*100, uas*100, pos*100))

    return las
end

local function test_model(feature_net, class_net, net, test_sentences, test_decisions, parser)
    local punct = load_punct(params.punct_set)

    test_feats(feature_net, class_net, test_sentences, test_decisions, parser, punct)

    print_evaluation(net, test_sentences, test_decisions, parser, punct, true)
end


--- Train ---
local function train_model(net, criterion, train_decisions, dev_sentences, dev_decisions, parser)
    local parameters, grad_params = net:getParameters()
    local batchsize = params.batch_size
    print(string.format("Using batchsize %d", batchsize))

    local batches = gen_batches(train_decisions, batchsize)
    local decisions_tensors, word_tensors, pos_tensors, label_tensors = get_batch_data(train_decisions)

    local last_accuracy = 0.0
    local best_acc = 0.0

    local punct = load_punct(params.punct_set)

    print(string.format("Training on %d examples (%d sentences)", num_train_examples, #train_decisions))

    for epoch = 1, params.num_epochs do
        net:training()

        -- randomly shuffle mini batches
        local num_batches = #batches
        local shuffle = torch.randperm(num_batches)
        local epoch_error = 0
        local startTime = sys.clock()
        local examples = 0
        local log_every = math.max(100, num_train_examples / 10)
        io.write('Starting epoch ', epoch, ' of ', params.num_epochs, '\n')
        for i = 1, num_batches
        do
            local idx = shuffle[i]
            local batch = batches[idx]
            local data = batch.data
            local class_labels = to_cuda(batch.label:long())

            examples = examples + class_labels:size(1)
            -- update function
            local function fEval(x)
                if parameters ~= x then parameters:copy(x) end
                net:zeroGradParameters()
                local output = net:forward(data)
                local err = criterion:forward(output, class_labels)
                local df_do = criterion:backward(output, class_labels)
                net:backward(data, df_do)
                epoch_error = epoch_error + err

                if params.l2 > 0 then grad_params:add(params.l2, parameters) end
                local grad_norm = grad_params:norm(2)
                if grad_norm > 1 then grad_params = grad_params:div(grad_norm) end

                return err, grad_params
            end

            -- update gradients
            optim[params.optim](fEval, parameters, opt_config, opt_state)

            -- not using modulo because examples jumps up every time and we might not
            -- hit exactly the number we want
            if(examples > log_every) then
                print(string.format("%20d examples at %5.2f examples/sec. Error: %5.5f",
                    examples, examples/(sys.clock() - startTime), (epoch_error / i)))
                log_every = log_every + log_every
            end
        end
        print(string.format('\nEpoch error = %f', epoch_error))
        if (epoch % params.evaluate_frequency == 0 or epoch == params.num_epochs) then
            net:evaluate()

            local accuracy = print_evaluation(net, criterion, dev_sentences, dev_decisions, parser, punct, false)

            -- end training early if accuracy goes down
            if params.stop_early and accuracy < last_accuracy then break else last_accuracy = accuracy end
            -- save the trained model if location specified
            if(accuracy > best_acc) then
                serialize_model(net, opt_state)
                best_acc = accuracy
            end
        end
    end
end


local net, criterion = build_net()
local parser = ProjShiftReduce(net, words_per_example, pos_per_example, labels_per_example, decision_domain_size, params.decision_map)
if(params.evaluate_only) then
    test_model(net, dev_sentences_portion, dev_decisions_portion, parser)
else
    train_model(net, criterion, train_decisions_portion, dev_sentences_portion, dev_decisions_portion, parser)
end

