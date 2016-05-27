require 'ParseState'
require 'torch'
require 'try-catch'

package.path = package.path .. ";src/main/lua/?.lua"

local ProjShiftReduce = torch.class('ProjShiftReduce')

-- constructor
function ProjShiftReduce:__init(net, num_word_feats, num_pos_feats, num_label_feats, num_decisions, decision2int)
    self.__index = self
    self.net = net
    self.num_word_feats = num_word_feats
    self.num_pos_feats = num_pos_feats
    self.num_label_feats = num_label_feats
    self.num_decisions = num_decisions
    self.int2decision = torch.zeros(num_decisions, 3)
    for line in io.lines(decision2int) do
        if(line ~= "") then
            local leftOrRightOrNo, shiftOrReduceOrPass, label, decision = string.match(line, "([^ ]+) ([^ ]+) ([^ ]+)\t([^\t]+)")
            self.int2decision[decision][1] = leftOrRightOrNo
            self.int2decision[decision][2] = shiftOrReduceOrPass
            self.int2decision[decision][3] = label
        end
    end
end

function ProjShiftReduce:predict(feats)
    local scores = self.net:forward(feats)
    local _, argmax = scores:max(2)
    return argmax[1][1]
end

function ProjShiftReduce:parse(sentence)
    local state = ParseState(1, 2, {}, sentence)
    while(state.input <= state.parseSentenceLength or state.stack > 1) do
        if(state.stack < 1) then self:shift(state)
        else
            -- make feats for prediction here
            local feats = self:compute_features_chen(state)
            local decision = self:predict(feats)
            local leftOrRightOrNo, shiftOrReduceOrPass, label = self:parse_decision(decision)
            if(leftOrRightOrNo ~= 0 and (state.input ~= state.parseSentenceLength+1 or not (shiftOrReduceOrPass == Constants.SHIFT and leftOrRightOrNo == Constants.NO))) then
            try {
                function()
                    self:transition(state, leftOrRightOrNo, shiftOrReduceOrPass, label)
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
    -- drop pretend root nodes and subtract 2
    return torch.add(state.headIndices:narrow(1, 2, state.parseSentenceLength-1), -2), state.arcLabels:narrow(1, 2, state.parseSentenceLength-1)
end

function ProjShiftReduce:parse_decision(decision)
    local decision_parts = self.int2decision[decision]
--    print(decision_parts[1], decision_parts[2], decision_parts[3])
    return decision_parts[1], decision_parts[2], decision_parts[3]
end

function ProjShiftReduce:transition(state, leftOrRightOrNo, shiftOrReduceOrPass, label)
    if(leftOrRightOrNo == Constants.LEFT and shiftOrReduceOrPass == Constants.REDUCE) then
        -- pop j, pop i, make arc j<-i, push j
        local j = state.stack
        self:passAux(state)
        local i = state.stack
        state:setHead(i, j, label)
        state.reducedIds[i] = true
        state.stack = j
    elseif (leftOrRightOrNo == Constants.RIGHT and shiftOrReduceOrPass ==  Constants.SHIFT) then
        -- pop j, make arc i<-j
        local j = state.stack
        self:passAux(state)
        local i = state.stack
        state:setHead(j, i, label)
        state.reducedIds[j] = true

    else self:shift(state)
    end
end

function ProjShiftReduce:passAux(state)
    local i = state.stack - 1
    while(i > 0) do
        if(not state.reducedIds[i]) then
            state.stack = i
            return
        end
        i = i - 1
    end
    state.stack = i
end

function ProjShiftReduce:shift(state)
    state.stack = state.input
    state.input = state.input + 1
end

-- Features (embeddings) for Chen & Manning 2014 (n=48):
-- - Top 3 words on stack and buffer (n=6)
-- - First and second left/rightmost children of top two words on the stack (n=8)
-- - Leftmost of leftmost / rightmost of rightmost children of top two words on the stack (n=4)
-- - POS tags of all of the above (n=18)
-- - Arc labels of left/rightmost children (n=12)
--
-- Sentence representation: ParseState refers to indices in sentence. The sentence
-- needs to tell me, at a given index: pos int value, label int value, word int value. so:
-- a sentence is a tensor of 4-tuples of ints: (1=word, 2=pos, 3=label, 4=head)
--
-- Net representation: this needs to return a table
function ProjShiftReduce:compute_features_chen(state)
    local word_feats = torch.Tensor(self.num_word_feats):fill(0)
    local pos_feats = torch.Tensor(self.num_pos_feats):fill(0)
    local label_feats = torch.Tensor(self.num_label_feats):fill(0)
    local word_feat_idx = 1
    local pos_feat_idx = 1
    local label_feat_idx = 1
    for i = 0,2 do
        -- Top 3 words on stack and buffer + their pos tags
        word_feats[word_feat_idx] = state:sentence(state:stackToken(-i))[1]; word_feat_idx = word_feat_idx + 1
        pos_feats[pos_feat_idx] = state:sentence(state:stackToken(-i))[2]; pos_feat_idx = pos_feat_idx + 1
        word_feats[word_feat_idx] = state:sentence(state:inputToken(i))[1]; word_feat_idx = word_feat_idx + 1
        pos_feats[pos_feat_idx] = state:sentence(state:inputToken(i))[2]; pos_feat_idx = pos_feat_idx + 1

        if(i < 2) then
            -- left/right and second left/rightmost children of top two words on stack, pos tags and arc labels
            local leftmostDep = state:leftmostDependent(state:stackToken(-i))
            word_feats[word_feat_idx] = state:sentence(leftmostDep)[1]; word_feat_idx = word_feat_idx + 1
            pos_feats[pos_feat_idx] = state:sentence(leftmostDep)[2]; pos_feat_idx = pos_feat_idx + 1
            if(leftmostDep == 0) then label_feats[label_feat_idx] = Constants.NULL_LABEL
            else label_feats[label_feat_idx] = state.arcLabels[leftmostDep]
            end; label_feat_idx = label_feat_idx + 1

            local rightmostDep = state:rightmostDependent(state:stackToken(-i))
            word_feats[word_feat_idx] = state:sentence(rightmostDep)[1]; word_feat_idx = word_feat_idx + 1
            pos_feats[pos_feat_idx] = state:sentence(rightmostDep)[2]; pos_feat_idx = pos_feat_idx + 1
            if(rightmostDep == 0) then label_feats[label_feat_idx] = Constants.NULL_LABEL
            else label_feats[label_feat_idx] = state.arcLabels[rightmostDep]
            end; label_feat_idx = label_feat_idx + 1

            local leftmostDep2 = state:leftmostDependent2(state:stackToken(-i))
            word_feats[word_feat_idx] = state:sentence(leftmostDep2)[1]; word_feat_idx = word_feat_idx + 1
            pos_feats[pos_feat_idx] = state:sentence(leftmostDep2)[2]; pos_feat_idx = pos_feat_idx + 1
            if(leftmostDep2 == 0) then label_feats[label_feat_idx] = Constants.NULL_LABEL
            else label_feats[label_feat_idx] = state.arcLabels[leftmostDep2]
            end; label_feat_idx = label_feat_idx + 1

            local rightmostDep2 = state:rightmostDependent2(state:stackToken(-i))
            word_feats[word_feat_idx] = state:sentence(rightmostDep2)[1]; word_feat_idx = word_feat_idx + 1
            pos_feats[pos_feat_idx] = state:sentence(rightmostDep2)[2]; pos_feat_idx = pos_feat_idx + 1
            if(rightmostDep2 == 0) then label_feats[label_feat_idx] = Constants.NULL_LABEL
            else label_feats[label_feat_idx] = state.arcLabels[rightmostDep2]
            end; label_feat_idx = label_feat_idx + 1

            local grandLeftmostDep = state:grandLeftmostDependent(state:stackToken(-i))
            word_feats[word_feat_idx] = state:sentence(grandLeftmostDep)[1]; word_feat_idx = word_feat_idx + 1
            pos_feats[pos_feat_idx] = state:sentence(grandLeftmostDep)[2]; pos_feat_idx = pos_feat_idx + 1
            if(grandLeftmostDep == 0) then label_feats[label_feat_idx] = Constants.NULL_LABEL
            else label_feats[label_feat_idx] = state.arcLabels[grandLeftmostDep]
            end; label_feat_idx = label_feat_idx + 1

            local grandRightmostDep = state:grandRightmostDependent(state:stackToken(-i))
            word_feats[word_feat_idx] = state:sentence(grandRightmostDep)[1]; word_feat_idx = word_feat_idx + 1
            pos_feats[pos_feat_idx] = state:sentence(grandRightmostDep)[2]; pos_feat_idx = pos_feat_idx + 1
            if(grandRightmostDep == 0) then label_feats[label_feat_idx] = Constants.NULL_LABEL
            else label_feats[label_feat_idx] = state.arcLabels[grandRightmostDep]
            end; label_feat_idx = label_feat_idx + 1
        end
    end
    if(word_feat_idx ~= 19) then print(string.format("Incorrect # word features computed! Want 18 have %d", word_feat_idx-1)) end
    if(pos_feat_idx ~= 19) then print(string.format("Incorrect # pos features computed! Want 18 have %d", pos_feat_idx-1)) end
    if(label_feat_idx ~= 13) then print(string.format("Incorrect # label features computed! Want 12 have %d", label_feat_idx-1)) end
    return {word_feats:view(1,self.num_word_feats), pos_feats:view(1,self.num_pos_feats), label_feats:view(1,self.num_label_feats)}
end