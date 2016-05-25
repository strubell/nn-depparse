package edu.umass.cs.iesl.nndepparse

object FeatureFunctionsFast {

  /*
    Features (embeddings) for Chen & Manning 2014 (n=48):
    - Top 3 words on stack and buffer (n=6)
    - First and second left/rightmost children of top two words on the stack (n=8)
    - Leftmost of leftmost / rightmost of rightmost children of top two words on the stack (n=4)
    - POS tags of all of the above (n=18)
    - Arc labels of left/rightmost children (n=12)
 */
  def computeChenFeatures(state: ParseState, wordFeats: Array[Int], posFeats: Array[Int], labelFeats: Array[Int]) = {
    var wordFeatIdx = 0
    var posFeatIdx = 0
    var labelFeatIdx = 0

    def stack(i: Int) = state.sentence(state.stackToken(i))
    def buffer(i: Int) = state.sentence(state.inputToken(i))
    def leftmostChild(i: Int) = state.sentence(state.leftmostDependent(state.stackToken(i)))
    def leftmostChild2(i: Int) = state.sentence(state.leftmostDependent2(state.stackToken(i)))
    def rightmostChild(i: Int) = state.sentence(state.rightmostDependent(state.stackToken(i)))
    def rightmostChild2(i: Int) = state.sentence(state.rightmostDependent2(state.stackToken(i)))
    def leftmostGrandchild(i: Int) = state.sentence(state.grandLeftmostDependent(state.stackToken(i)))
    def rightmostGrandchild(i: Int) = state.sentence(state.grandRightmostDependent(state.stackToken(i)))

    (0 until 3).foreach{i =>
      // Top 3 words on stack and buffer + their pos tags
      wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${stack(-i).string}", 1); wordFeatIdx += 1
      posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${stack(-i).posTagString}");  posFeatIdx += 1
      wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${buffer(i).string}", 1); wordFeatIdx += 1
      posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${buffer(i).posTagString}");  posFeatIdx += 1

      if(i < 2){
        // left/right and second left/rightmost children of top two words on stack, pos tags and arc labels
        val leftmostDep = state.leftmostDependent(state.stackToken(-i))
        wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${leftmostChild(-i).string}", 1);  wordFeatIdx += 1
        posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${leftmostChild(-i).posTagString}");  posFeatIdx += 1
        labelFeats(labelFeatIdx) = IntMaps.labelToIntMap(s"${if(leftmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(leftmostDep)}"); labelFeatIdx += 1

        val rightmostDep = state.rightmostDependent(state.stackToken(-i))
        wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${rightmostChild(-i).string}", 1);  wordFeatIdx += 1
        posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${rightmostChild(-i).posTagString}");  posFeatIdx += 1
        labelFeats(labelFeatIdx) = IntMaps.labelToIntMap(s"${if(rightmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(rightmostDep)}"); labelFeatIdx += 1

        val leftmostDep2 = state.leftmostDependent2(state.stackToken(-i))
        wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${leftmostChild2(-i).string}", 1);  wordFeatIdx += 1
        posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${leftmostChild2(-i).posTagString}");  posFeatIdx += 1
        labelFeats(labelFeatIdx) = IntMaps.labelToIntMap(s"${if(leftmostDep2 == -1) ParserConstants.NULL_STRING else state.arcLabels(leftmostDep2)}"); labelFeatIdx += 1

        val rightmostDep2 = state.rightmostDependent2(state.stackToken(-i))
        wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${rightmostChild2(-i).string}", 1);  wordFeatIdx += 1
        posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${rightmostChild2(-i).posTagString}");  posFeatIdx += 1
        labelFeats(labelFeatIdx) = IntMaps.labelToIntMap(s"${if(rightmostDep2 == -1) ParserConstants.NULL_STRING else state.arcLabels(rightmostDep2)}"); labelFeatIdx += 1

        val grandLeftmostDep = state.grandLeftmostDependent(state.stackToken(-i))
        wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${leftmostGrandchild(-i).string}", 1);  wordFeatIdx += 1
        posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${leftmostGrandchild(-i).posTagString}");  posFeatIdx += 1
        labelFeats(labelFeatIdx) = IntMaps.labelToIntMap(s"${if(grandLeftmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(grandLeftmostDep)}"); labelFeatIdx += 1

        val grandRightmostDep = state.grandRightmostDependent(state.stackToken(-i))
        wordFeats(wordFeatIdx) = IntMaps.wordToIntMap.getOrElse(s"${rightmostGrandchild(-i).string}", 1);  wordFeatIdx += 1
        posFeats(posFeatIdx) = IntMaps.posToIntMap(s"${rightmostGrandchild(-i).posTagString}");  posFeatIdx += 1
        labelFeats(labelFeatIdx) = IntMaps.labelToIntMap(s"${if(grandRightmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(grandRightmostDep)}"); labelFeatIdx += 1
      }
    }
  }
}
