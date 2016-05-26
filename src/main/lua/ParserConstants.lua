package.path = package.path .. ";src/main/lua/?.lua"

-- this is a lua "static class"
local ParserConstants = torch.class('ParserConstants')

function ParserConstants:__init(pos2int, label2int, word2int)

    -- default value, for when we have pos tags
    self.NONE_POS = 0
    self.NULL_WORD = 0
    self.ROOT_WORD = 0
    self.NULL_POS = 0
    self.ROOT_POS = 0
    self.NULL_LABEL = 0

    -- read in pos2int table, set pos <NULL> and <ROOT>
    local file = io.open(pos2int)
    local found = 4
    local line = file:read()
    local line_num = 1
    while(found > 0 and line ~= nil) do
        if(line ~= "") then
            local pos_str, pos_int = string.match(line, "([^\t]+)\t([^\t]+)")
            if(pos_str == "<NULL>") then
                self.NULL_POS = tonumber(pos_int)
                found = found - 1
            elseif(pos_str == "<ROOT>") then
                self.ROOT_POS = tonumber(pos_int)
                found = found - 1
            elseif(pos_str == "<NONE>") then
                self.NONE_POS = tonumber(pos_int)
                found = found - 1
            elseif(pos_str == "XX") then
                self.NO_POS = tonumber(pos_int)
                found = found - 1
            end
        end
        line_num = line_num + 1
        line = file:read()
    end
    file:close()
    if(self.NULL_POS == 0) then self.NULL_POS = line_num; line_num = line_num + 1 end
    if(self.ROOT_POS == 0) then self.ROOT_POS = line_num end

    file = io.open(label2int)
    found = 1
    line = file:read()
    line_num = 1
    while(found > 0 and line ~= nil) do
        if(line ~= "") then
            local label_str, label_int = string.match(line, "([^\t]+)\t([^\t]+)")
            if(label_str == "<NULL>") then
                self.NULL_LABEL = tonumber(label_int)
                found = found - 1
            end
        end
        line_num = line_num + 1
        line = file:read()
    end
    file:close()
    if(self.NULL_LABEL == 0) then self.NULL_LABEL = line_num end


    file = io.open(word2int)
    found = 2
    line = file:read()
    line_num = 1
    while(found > 0 and line ~= nil) do
        if(line ~= "") then
            local word_str, word_int = string.match(line, "([^\t]+)\t([^\t]+)")
--            print(word_str, word_int)
            if(word_str == "<NULL>") then
                self.NULL_WORD = tonumber(word_int)
                found = found - 1
            elseif(word_str == "<ROOT>") then
                self.ROOT_WORD = tonumber(word_int)
                found = found - 1
            end
        end
        line_num = line_num + 1
        line = file:read()
    end
    file:close()
    if(self.NULL_WORD == 0) then self.NULL_WORD = line_num; line_num = line_num + 1 end
    if(self.ROOT_WORD == 0) then self.ROOT_WORD = line_num end


    self.SHIFT = 1
    self.REDUCE = 2
    self.PASS = 3
    self.LEFT = 4
    self.RIGHT = 5
    self.NO = 6

    self.NULL = torch.Tensor(1,9)
    self.NULL[1][1] = self.NULL_WORD -- word
    self.NULL[1][2] = self.NULL_POS -- pos
    self.NULL[1][3] = self.NULL_LABEL -- label

    self.ROOT = torch.Tensor(1,9)
    self.ROOT[1][1] = self.ROOT_WORD -- word
    self.ROOT[1][2] = self.ROOT_POS -- pos
end

