
require 'torch'
cmd = torch.CmdLine()
cmd:option('-inFile','','input file')
cmd:option('-outFile','','out')
cmd:option('-feats', 'chen', "Type of features we're loading")
local params = cmd:parse(arg)

print(params)

local line_regex = "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)"
local line_regex_no_decision = "[^\t]+\t([^\t]+)\t([^\t]+)\t([^\t]+)"

-- get feature counts from first line
local in_file = io.open(params.inFile, "r")
in_file:read()
local line = in_file:read()
in_file:close()

local word_feats, pos_feats, label_feats = string.match(line, line_regex_no_decision)

local num_word_feats = 0
local num_pos_feats = 0
local num_label_feats = 0
for _ in string.gmatch(word_feats, "%S+") do
    num_word_feats = num_word_feats + 1
end
for _ in string.gmatch(pos_feats, "%S+") do
    num_pos_feats = num_pos_feats + 1
end
for _ in string.gmatch(label_feats, "%S+") do
    num_label_feats = num_label_feats + 1
end

print(string.format("Loading %d word feats, %d pos feats, %d label feats", num_word_feats, num_pos_feats, num_label_feats))

local tensors = {}
local sentence_lines = {}
local word_domain_size = 0
local pos_domain_size = 0
local label_domain_size = 0
local decision_domain_size = 0
for line in io.lines(params.inFile) do
    if(line == "" and #sentence_lines > 0) then
        local num_lines = #sentence_lines
        local decisions = torch.Tensor(num_lines)
        local word_data = torch.Tensor(num_lines, num_word_feats)
        local pos_data = torch.Tensor(num_lines, num_pos_feats)
        local label_data = torch.Tensor(num_lines, num_label_feats)
        for line_idx=1,num_lines do
            local sentence_line = sentence_lines[line_idx]
            local decision, word_feats, pos_feats, label_feats = string.match(sentence_line, line_regex)
            local int_decision = tonumber(decision)
            decisions[line_idx] = int_decision
            if(int_decision > decision_domain_size) then decision_domain_size = int_decision end
            local feat_idx = 1
            -- load word feats
            for feat in string.gmatch(word_feats, "%S+") do
                local int_feat = tonumber(feat)
                word_data[line_idx][feat_idx] = int_feat
                if(int_feat > word_domain_size) then word_domain_size = int_feat end
                feat_idx = feat_idx + 1
            end
            -- load pos feats
            feat_idx = 1
            for feat in string.gmatch(pos_feats, "%S+") do
                local int_feat = tonumber(feat)
                pos_data[line_idx][feat_idx] = int_feat
                if(int_feat > pos_domain_size) then pos_domain_size = int_feat end
                feat_idx = feat_idx + 1
            end
            -- load label feats
            feat_idx = 1
            for feat in string.gmatch(label_feats, "%S+") do
                local int_feat = tonumber(feat)
                label_data[line_idx][feat_idx] = int_feat
                if(int_feat > label_domain_size) then label_domain_size = int_feat end
                feat_idx = feat_idx + 1
            end
        end
        table.insert(tensors, {decisions=decisions, word_feats=word_data, pos_feats=pos_data, label_feats=label_data})
        sentence_lines = {}
    elseif(line ~= "") then
        table.insert(sentence_lines, line)
    end
end

-- don't forget last sentence
local num_lines = #sentence_lines
if(num_lines > 0) then
    local num_lines = #sentence_lines
    local decisions = torch.Tensor(num_lines)
    local word_data = torch.Tensor(num_lines, num_word_feats)
    local pos_data = torch.Tensor(num_lines, num_pos_feats)
    local label_data = torch.Tensor(num_lines, num_label_feats)
    for line_idx=1,num_lines do
        local sentence_line = sentence_lines[line_idx]
        local decision, word_feats, pos_feats, label_feats = string.match(sentence_line, line_regex)
        local int_decision = tonumber(decision)
        decisions[line_idx] = int_decision
        if(int_decision > decision_domain_size) then decision_domain_size = int_decision end
        local feat_idx = 1
        -- load word feats
        for feat in string.gmatch(word_feats, "%S+") do
            local int_feat = tonumber(feat)
            word_data[line_idx][feat_idx] = int_feat
            if(int_feat > word_domain_size) then word_domain_size = int_feat end
            feat_idx = feat_idx + 1
        end
        -- load pos feats
        feat_idx = 1
        for feat in string.gmatch(pos_feats, "%S+") do
            local int_feat = tonumber(feat)
            pos_data[line_idx][feat_idx] = int_feat
            if(int_feat > pos_domain_size) then pos_domain_size = int_feat end
            feat_idx = feat_idx + 1
        end
        -- load label feats
        feat_idx = 1
        for feat in string.gmatch(label_feats, "%S+") do
            local int_feat = tonumber(feat)
            label_data[line_idx][feat_idx] = int_feat
            if(int_feat > label_domain_size) then label_domain_size = int_feat end
            feat_idx = feat_idx + 1
        end
    end
    table.insert(tensors, { decisions=decisions, word_feats=word_data, pos_feats=pos_data, label_feats=label_data})
end

tensors.decision_domain_size = decision_domain_size
tensors.word_domain_size = word_domain_size
tensors.pos_domain_size = pos_domain_size
tensors.label_domain_size = label_domain_size

tensors.num_word_feats = num_word_feats
tensors.num_pos_feats = num_pos_feats
tensors.num_label_feats = num_label_feats

torch.save(params.outFile, tensors)
