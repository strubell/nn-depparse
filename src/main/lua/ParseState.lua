require 'torch'
require 'try-catch'

package.path = package.path .. ";src/main/lua/?.lua"

local ParseState = torch.class('ParseState')

-- constructor
function ParseState:__init(stack, input, reducedIds, sentence, pos_features)
    self.__index = self
    self.parseSentenceLength = sentence:size(1) + 1
    self.stack = stack
    self.input = input
    self.reducedIds = reducedIds
    self._sentence = sentence
    self.headIndices = torch.Tensor(self.parseSentenceLength):fill(0)
    self.arcLabels = torch.Tensor(self.parseSentenceLength):fill(0)

    self.leftmostDeps = torch.Tensor(self.parseSentenceLength):fill(0)
    self.rightmostDeps = torch.Tensor(self.parseSentenceLength):fill(0)
    self.leftmostDeps2 = torch.Tensor(self.parseSentenceLength):fill(0)
    self.rightmostDeps2 = torch.Tensor(self.parseSentenceLength):fill(0)

    -- todo should maybe actually be a root representation
    self.features = pos_features

end

function ParseState:print()
    print("sentence length: ", self.parseSentenceLength)
    print(string.format("stack: %d stack-1: %d input: %d", self.stack, self:stackToken(-1), self.input))
    print(string.format("reduced ids: %s", self:mk_string(self.reducedIds, " ")))
    print("arc labels: ")
    print(self:mk_string_tensor(self.arcLabels, " "))
    print("head indices: ")
    print(self:mk_string_tensor(self.headIndices, " "))

    print("true arc labels:")
    print(self:mk_string_tensor(self._sentence:select(2,3), " "))
    print("true heads:")
    print(self:mk_string_tensor(torch.add(self._sentence:select(2,4), 2), " "))


    print("leftmost deps: ")
    print(self:mk_string_tensor(self.leftmostDeps, " "))
    print("rightmost deps: ")
    print(self:mk_string_tensor(self.rightmostDeps, " "))
end

function ParseState:mk_string(t, sep)
    local str = ""
    if(t) then
        for i,_ in pairs(t) do
            str = str .. i .. sep
        end
    end
    return str
end

function ParseState:mk_string_tensor(tensor, sep)
    local str = ""
    if(tensor) then
        for i = 1,tensor:size()[1] do
            str = str .. tensor[i] .. sep
        end
    end
    return str
end

function ParseState:sentence(index)
    if(index == 1) then
        return Constants.ROOT[1]
    elseif (index > 1 and index <= self.parseSentenceLength) then
        return self._sentence[index-1]
    else
        return Constants.NULL[1]
    end
end

function ParseState:setHead(tokenIndex, headIndex, label)

        self.headIndices[tokenIndex] = headIndex
        self.arcLabels[tokenIndex] = label

        if(headIndex > 0) then
            if(tokenIndex < self.leftmostDeps[headIndex] or self.leftmostDeps[headIndex] == 0) then
                self.leftmostDeps2[headIndex] = self.leftmostDeps[headIndex]
                self.leftmostDeps[headIndex] = tokenIndex
            end
            if(tokenIndex > self.rightmostDeps[headIndex] or self.rightmostDeps[headIndex] == 0) then
                self.rightmostDeps2[headIndex] = self.rightmostDeps[headIndex]
                self.rightmostDeps[headIndex] = tokenIndex
            end
        end
--    end
end

function ParseState:isDescendantOf(firstIndex, secondIndex)
    local firstHeadIndex = self.headIndices[firstIndex]
    if(firstHeadIndex == 0) then return false -- firstIndex has no head, so it can't be a descendant
    elseif(firstHeadIndex == secondIndex) then return true
    else return self:isDescendantOf(firstHeadIndex, secondIndex)
    end
end

function ParseState:leftmostDependent(tokenIndex)
    if(tokenIndex <= 0 or tokenIndex >= self.parseSentenceLength) then return 0
    else return self.leftmostDeps[tokenIndex]
    end
end

function ParseState:rightmostDependent(tokenIndex)
    if(tokenIndex <= 0 or tokenIndex >= self.parseSentenceLength) then return 0
    else return self.rightmostDeps[tokenIndex]
    end
end

function ParseState:leftmostDependent2(tokenIndex)
    if(tokenIndex <= 0 or tokenIndex >= self.parseSentenceLength) then return 0
    else return self.leftmostDeps2[tokenIndex]
    end
end

function ParseState:rightmostDependent2(tokenIndex)
    if(tokenIndex <= 0 or tokenIndex >= self.parseSentenceLength) then return 0
    else return self.rightmostDeps2[tokenIndex]
    end
end

function ParseState:grandLeftmostDependent(tokenIndex)
    if(tokenIndex <= 0 or tokenIndex >= self.parseSentenceLength) then return 0
    else
        local i = self.leftmostDeps[tokenIndex]
        if(i == 0) then return 0
        else return self.leftmostDeps[i]
        end
    end
end

function ParseState:grandRightmostDependent(tokenIndex)
    if(tokenIndex <= 0 or tokenIndex >= self.parseSentenceLength) then return 0
    else
        local i = self.rightmostDeps[tokenIndex]
        if(i == 0) then return 0
        else return self.rightmostDeps[i]
        end
    end
end

function ParseState:leftNearestSibling(tokenIndex)
    local tokenHeadIndex = self.headIndices[tokenIndex]
    if(tokenHeadIndex ~= 0) then
        local i = tokenIndex - 1
        while(i > 0) do
            if(self.headIndices[i] ~= 0 and self.headIndices[i] == tokenHeadIndex) then
                return i
            end
            i = i - 1
        end
    end
    return 0
end

function ParseState:rightNearestSibling(tokenIndex)
    local tokenHeadIndex = self.headIndices[tokenIndex]
    if(tokenHeadIndex ~= 0) then
        local i = tokenIndex + 1
        while(i <= self.parseSentenceLength) do
            if(self.headIndices[i] ~= 0 and self.headIndices[i] == tokenHeadIndex) then
                return i
            end
            i = i + 1
        end
    end
    return 0
end

function ParseState:inputToken(offset)
    local i = self.input + offset
    if(i < 1 or self.parseSentenceLength < i) then return 0
    else return i
    end
end

function ParseState:lambdaToken(offset)
    local i = self.stack + offset
    if(i < 1 or self.parseSentenceLength < i) then return 0
    else return i
    end
end

function ParseState:stackToken(offset)
    if(offset == 0) then return self.stack end
    local off = math.abs(offset)
    local dir = 1
    if(offset < 0) then dir = -1 end
    local i = self.stack + dir
    while(1 < i and i < self.input) do
        if(not self.reducedIds[i]) then
            off = off - 1
            if(off == 0) then return i end
        end
        i = i + dir
    end
    return i
end