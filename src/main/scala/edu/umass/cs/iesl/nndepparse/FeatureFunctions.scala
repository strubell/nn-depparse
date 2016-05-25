package edu.umass.cs.iesl.nndepparse

import scala.collection.mutable.ArrayBuffer

object FeatureFunctions{

  /*
    Features (embeddings) for Chen & Manning 2014 (n=48):
    - Top 3 words on stack and buffer (n=6)
    - First and second left/rightmost children of top two words on the stack (n=8)
    - Leftmost of leftmost / rightmost of rightmost children of top two words on the stack (n=4)
    - POS tags of all of the above (n=18)
    - Arc labels of left/rightmost children (n=12)
 */
  def computeChenFeatures(state: ParseState): Array[Array[String]] = {
    val wordFeats = ArrayBuffer[String]()
    val posFeats = ArrayBuffer[String]()
    val labelFeats = ArrayBuffer[String]()

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
      wordFeats += s"word@s$i=${stack(-i).string}"
      posFeats += s"pos@s$i=${stack(-i).posTagString}"
      wordFeats += s"word@b$i=${buffer(i).string}"
      posFeats += s"pos@b$i=${buffer(i).posTagString}"

      if(i < 2){
        // left/right and second left/rightmost children of top two words on stack, pos tags and arc labels
        val leftmostDep = state.leftmostDependent(state.stackToken(-i))
        wordFeats += s"leftword@s$i=${leftmostChild(-i).string}"
        posFeats += s"leftpos@s$i=${leftmostChild(-i).posTagString}"
        labelFeats += s"leftlabel@s$i=${if(leftmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(leftmostDep)}"

        val rightmostDep = state.rightmostDependent(state.stackToken(-i))
        wordFeats += s"rightword@s$i=${rightmostChild(-i).string}"
        posFeats += s"rightpos@s$i=${rightmostChild(-i).posTagString}"
        labelFeats += s"rightlabel@s$i=${if(rightmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(rightmostDep)}"

        val leftmostDep2 = state.leftmostDependent2(state.stackToken(-i))
        wordFeats += s"leftword2@s$i=${leftmostChild2(-i).string}"
        posFeats += s"leftpos2@s$i=${leftmostChild2(-i).posTagString}"
        labelFeats += s"leftlabel2@s$i=${if(leftmostDep2 == -1) ParserConstants.NULL_STRING else state.arcLabels(leftmostDep2)}"

        val rightmostDep2 = state.rightmostDependent2(state.stackToken(-i))
        wordFeats += s"rightword2@s$i=${rightmostChild2(-i).string}"
        posFeats += s"rightpos2@s$i=${rightmostChild2(-i).posTagString}"
        labelFeats += s"rightlabel2@s$i=${if(rightmostDep2 == -1) ParserConstants.NULL_STRING else state.arcLabels(rightmostDep2)}"

        val grandLeftmostDep = state.grandLeftmostDependent(state.stackToken(-i))
        wordFeats += s"grandleftword@s$i=${leftmostGrandchild(-i).string}"
        posFeats += s"grandleftpos@s$i=${leftmostGrandchild(-i).posTagString}"
        labelFeats += s"grandleftlabel@s$i=${if(grandLeftmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(grandLeftmostDep)}"

        val grandRightmostDep = state.grandRightmostDependent(state.stackToken(-i))
        wordFeats += s"grandrightword@s$i=${rightmostGrandchild(-i).string}"
        posFeats += s"grandrightpos@s$i=${rightmostGrandchild(-i).posTagString}"
        labelFeats += s"grandrightlabel@s$i=${if(grandRightmostDep == -1) ParserConstants.NULL_STRING else state.arcLabels(grandRightmostDep)}"
      }
    }
    assert(wordFeats.length == 18, s"Expected to compute 18 word feats, got ${wordFeats.length}: ${wordFeats.mkString(" ")}")
    assert(posFeats.length == 18, s"Expected to compute 18 pos feats, got ${posFeats.length}: ${posFeats.mkString(" ")}")
    assert(labelFeats.length == 12, s"Expected to compute 12 label feats, got ${labelFeats.length}: ${labelFeats.mkString(" ")}")
//    wordFeats.mkString(" ") + "\t" + posFeats.mkString(" ") + "\t" + labelFeats.mkString(" ")  + (if(useShapeFeats) "\t" + shapeFeats.mkString(" ") else "")
    Array(wordFeats.toArray, posFeats.toArray, labelFeats.toArray)

  }
}
