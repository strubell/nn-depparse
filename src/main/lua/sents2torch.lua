
require 'torch'
cmd = torch.CmdLine()
cmd:option('-inFile','','input file')
cmd:option('-outFile','','out')
local params = cmd:parse(arg)

print(params)

local line_regex = "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)"

-- todo generate regex using this number
local num_fields = 5

local word_domain_size = 0
local pos_domain_size = 0
local label_domain_size = 0

local tensors = {}
local sentence_lines = {}
for line in io.lines(params.inFile) do
    if(line == "" and #sentence_lines > 0) then
        local num_tokens = #sentence_lines
        local sentence = torch.Tensor(num_tokens, num_fields)
        for i=1,#sentence_lines do
            local sentence_line = sentence_lines[i]
            local word, pos, label, head, gold_pos = string.match(sentence_line, line_regex)
            sentence[i][1] = tonumber(word)
            sentence[i][2] = tonumber(pos)
            sentence[i][3] = tonumber(label)
            sentence[i][4] = tonumber(head)
            sentence[i][5] = tonumber(gold_pos)
            if(sentence[i][1] > word_domain_size) then word_domain_size = sentence[i][1] end
            if(sentence[i][2] > pos_domain_size) then pos_domain_size = sentence[i][2] end
            if(sentence[i][3] > label_domain_size) then label_domain_size = sentence[i][3] end
            if(sentence[i][5] > pos_domain_size) then pos_domain_size = sentence[i][5] end
        end
        table.insert(tensors, sentence)
        sentence_lines = {}
    elseif(line ~= "") then
        table.insert(sentence_lines, line)
    end
end

-- don't forget last sentence
local num_tokens = #sentence_lines
if(num_tokens > 0) then
    local sentence = torch.Tensor(num_tokens, num_fields)
    for i=1,#sentence_lines do
        local sentence_line = sentence_lines[i]
        local word, pos, label, head, gold_pos = string.match(sentence_line, line_regex)
        sentence[i][1] = tonumber(word)
        sentence[i][2] = tonumber(pos)
        sentence[i][3] = tonumber(label)
        sentence[i][4] = tonumber(head)
        sentence[i][5] = tonumber(gold_pos)
        if(sentence[i][1] > word_domain_size) then word_domain_size = sentence[i][1] end
        if(sentence[i][2] > pos_domain_size) then pos_domain_size = sentence[i][2] end
        if(sentence[i][3] > label_domain_size) then label_domain_size = sentence[i][3] end
        if(sentence[i][5] > pos_domain_size) then pos_domain_size = sentence[i][5] end
    end
    table.insert(tensors, sentence)
end

tensors.word_domain_size = word_domain_size
tensors.pos_domain_size = pos_domain_size
tensors.label_domain_size = label_domain_size

torch.save(params.outFile, tensors)
