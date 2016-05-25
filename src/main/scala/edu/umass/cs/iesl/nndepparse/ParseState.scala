package edu.umass.cs.iesl.nndepparse

import scala.annotation.tailrec
import scala.collection.mutable.Set

/* Represents current stack, parse, input during transition-based parsing, super mutable */
class ParseState(var stack: Int, var input: Int, val reducedIds: Set[Int], val sentence: LightweightParseSentence) {
  val parseSentenceLength = sentence.length

  val headIndices = Array.fill[Int](parseSentenceLength)(-1)
  val arcLabels = Array.fill[String](parseSentenceLength)("")

  val leftmostDeps = Array.fill[Int](parseSentenceLength)(-1)
  val rightmostDeps = Array.fill[Int](parseSentenceLength)(-1)

  val leftmostDeps2 = Array.fill[Int](parseSentenceLength)(-1)
  val rightmostDeps2 = Array.fill[Int](parseSentenceLength)(-1)

  def goldHeads = sentence.goldHeads
  def goldLabels = sentence.goldLabels

  def setHead(tokenIndex: Int, headIndex: Int, label: String) = {
    // set head
    headIndices(tokenIndex) = headIndex
    arcLabels(tokenIndex) = label

    // update left and rightmost dependents
    if(headIndex != -1){
      if (tokenIndex < leftmostDeps(headIndex) || leftmostDeps(headIndex) == -1) {
        leftmostDeps2(headIndex) = leftmostDeps(headIndex)
        leftmostDeps(headIndex) = tokenIndex
      }
      if(tokenIndex > rightmostDeps(headIndex) || rightmostDeps(headIndex) == -1) {
        rightmostDeps2(headIndex) = rightmostDeps(headIndex)
        rightmostDeps(headIndex) = tokenIndex
      }
    }
  }

  @tailrec final def isDescendantOf(firstIndex: Int, secondIndex: Int): Boolean = {
    val firstHeadIndex = headIndices(firstIndex)
    if (firstHeadIndex == -1) false // firstIndex has no head, so it can't be a descendant
    else if (firstHeadIndex == secondIndex) true
    else isDescendantOf(firstHeadIndex, secondIndex)
  }

  def leftmostDependent(tokenIndex: Int): Int = {
    if (tokenIndex == -1) -1
    else leftmostDeps(tokenIndex)
  }

  def rightmostDependent(tokenIndex: Int): Int = {
    if (tokenIndex == -1) -1
    else rightmostDeps(tokenIndex)
  }

  def leftmostDependent2(tokenIndex: Int): Int = {
    if (tokenIndex == -1) -1
    else leftmostDeps2(tokenIndex)
  }

  def rightmostDependent2(tokenIndex: Int): Int = {
    if (tokenIndex == -1) -1
    else rightmostDeps2(tokenIndex)
  }

  def grandLeftmostDependent(tokenIndex: Int): Int = {
    if (tokenIndex == -1) -1
    else{
      val i = leftmostDeps(tokenIndex)
      if (i == -1) -1
      else leftmostDeps(i)
    }
  }

  def grandRightmostDependent(tokenIndex: Int): Int = {
    if (tokenIndex == -1) -1
    else {
      val i = rightmostDeps(tokenIndex)
      if (i == -1) -1
      else rightmostDeps(i)
    }
  }

  def leftNearestSibling(tokenIndex: Int): Int = {
    val tokenHeadIndex = headIndices(tokenIndex)
    if(tokenHeadIndex != -1){
      var i = tokenIndex - 1
      while(i >= 0){
        if (headIndices(i) != -1 && headIndices(i) == tokenHeadIndex)
          return i
        i -= 1
      }
    }
    -1
  }

  def rightNearestSibling(tokenIndex: Int): Int = {
    val tokenHeadIndex = headIndices(tokenIndex)
    if(tokenHeadIndex != -1){
      var i = tokenIndex + 1
      while(i < parseSentenceLength){
        if(headIndices(i) != -1 && headIndices(i) == tokenHeadIndex)
          return i
        i += 1
      }
    }
    -1
  }

  def inputToken(offset: Int): Int = {
    val i = input + offset
    if (i < 0 || parseSentenceLength - 1 < i) -1
    else i
  }

  def lambdaToken(offset: Int): Int = {
    val i = stack + offset
    if (i < 0 || parseSentenceLength - 1 < i) -1
    else i
  }

  def stackToken(offset: Int): Int = {
    if (offset == 0)
      return stack
    var off = math.abs(offset)
    var dir = if (offset < 0) -1 else 1
    var i = stack + dir
    while (0 < i && i < input) {
      if (!reducedIds.contains(i)) {
        off -= 1
        if (off == 0)
          return i
      }
      i += dir
    }
    i
  }

  def print() = {
    println(s"heads: ${headIndices.mkString(" ")}")
    println(s"labels: ${arcLabels.map(l => if(l == "") "-1" else l).mkString(" ")}")
    println(s"gold heads: ${goldHeads.mkString(" ")}")
    println(s"gold labels: ${goldLabels.mkString(" ")}")
    println(s"gold pos: ${(0 until parseSentenceLength).map(i => sentence(i).posTagString).mkString(" ")}")
    println(s"stack: $stack stack-1: ${stackToken(-1)} input: $input")
    println(s"reduced ids: ${reducedIds.mkString(" ")}")
  }
}
