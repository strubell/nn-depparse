package edu.umass.cs.iesl.nndepparse

import cc.factorie.util.JavaHashSet
import cc.factorie.variable.CategoricalDomain

import scala.collection.mutable.ArrayBuffer

object ParseDecisionDomain extends CategoricalDomain[String]{
  import ParserConstants._
  val defaultLabel = StanfordParseTreeLabelDomain.defaultCategory
  val defaultCategory = NOTHING + " " + NOTHING + " " + defaultLabel
  this += defaultCategory
}

class ParseDecision(val leftOrRightOrNo: Int, val shiftOrReduceOrPass: Int, val label: String){
  override def toString = leftOrRightOrNo + " " + shiftOrReduceOrPass + " " + StanfordParseTreeLabelDomain.index(label)
  def readableString = s"${ParserConstants(leftOrRightOrNo)} ${ParserConstants(shiftOrReduceOrPass)} $label"
}

object ProjectiveArcStandardShiftReduce {
  import ParserConstants._

  def getGoldDecisionFeatures(s: LightweightParseSentence, featureFunction: ParseState => Array[Array[String]]): Array[String] = {
    val state = new ParseState(0, 1, JavaHashSet[Int](), s)
    val decisions = new ArrayBuffer[String]()
    while(state.input < state.parseSentenceLength || state.stack > 0) {
      if (state.stack < 0)
        shift(state)
      else {
        val label = getGoldDecision(state)
        if(state.input != state.parseSentenceLength || !(label.shiftOrReduceOrPass == ParserConstants.SHIFT && label.leftOrRightOrNo == ParserConstants.NO)) {
          decisions += label.toString + "\t" + featureFunction(state).map(_.mkString(" ")).mkString("\t")
          transition(state, label)
        }
        else{
          state.stack = 0
          state.input = state.input + 1
        }
      }
    }
    decisions.toArray
  }

  def getGoldDecisions(s: LightweightParseSentence): Array[(ParseState, ParseDecision)] = {
    val state = new ParseState(0, 1, JavaHashSet[Int](), s)
    val decisions = new ArrayBuffer[(ParseState, ParseDecision)]()
    while(state.input < state.parseSentenceLength || state.stack > 0) {
      if (state.stack < 0)
        shift(state)
      else {
        val label = getGoldDecision(state)
        if(state.input != state.parseSentenceLength || !(label.shiftOrReduceOrPass == ParserConstants.SHIFT && label.leftOrRightOrNo == ParserConstants.NO)) {
          decisions += ((state, label))
          transition(state, label)
        }
        else{
          state.stack = 0
          state.input = state.input + 1
        }
      }
    }
    decisions.toArray
  }

  def parse(s: LightweightParseSentence, predict: ParseState => String): (Array[Int], Array[String]) = {
    val state = new ParseState(0, 1, JavaHashSet[Int](), s)
    while(state.input < state.parseSentenceLength || state.stack > 0) {
      if (state.stack < 0)
        shift(state)
      else {
        val prediction = predict(state)
        val split = prediction.split(" ")
        // todo slow
        val label = new ParseDecision(split(0).toInt, split(1).toInt, IntMaps.intToLabelMap(split(2).toInt))
        if(state.input != state.parseSentenceLength || !(label.shiftOrReduceOrPass == ParserConstants.SHIFT && label.leftOrRightOrNo == ParserConstants.NO)) {
          transition(state, label)
        }
        else{
          state.stack = 0
          state.input = state.input + 1
        }
      }
    }
    (state.headIndices, state.arcLabels)
  }

  def transitionOnGoldLabel(state: ParseState) = {
    val label = getGoldDecision(state)
    transition(state, label)
  }

  def transition(state: ParseState, label: ParseDecision) = {
    // arc-standard
    if (label.leftOrRightOrNo == LEFT && label.shiftOrReduceOrPass == REDUCE) {
      /* i=second, j=first elements on stack; make arc i->j, pop j, i, push j back on */
      val j = state.stack
      passAux(state)
      val i = state.stack
      state.setHead(i, j, label.label) // makes an arc with j as the head
      state.reducedIds.add(i)
      state.stack = j
    }
    else if (label.leftOrRightOrNo == RIGHT && label.shiftOrReduceOrPass == SHIFT) {
      /* i=second, j=first elements on stack; make arc i<-j, pop j */
      val j = state.stack
      passAux(state)
      val i = state.stack
      state.setHead(j, i, label.label) // makes an arc with i as the head
      state.reducedIds.add(j)
    }
    else {
      // must be shift
      shift(state)
    }
  }

  /* This is really just "pop" */
  private def passAux(state: ParseState): Unit = {
    var i = state.stack - 1
    while (i >= 0) {
      if (!state.reducedIds.contains(i)) {
        state.stack = i
        return
      }
      i -= 1
    }
    state.stack = i
  }

  private def leftArc(label: String, state: ParseState)  { state.setHead(state.stack, state.input, label) }
  private def rightArc(label: String, state: ParseState) { state.setHead(state.input, state.stack, label) }
  def shift(state: ParseState)  { state.stack = state.input; state.input += 1 }
  private def reduce(state: ParseState) { state.reducedIds.add(state.stack); passAux(state) }

  def getGoldDecision(state: ParseState): ParseDecision = {
    val goldLRN = getGoldLRN(state)
    val shiftOrReduceOrPass =
      goldLRN match {
        case LEFT  => REDUCE
        case RIGHT => SHIFT
        case _ => if(shouldGoldReduce(hasHead=false, state=state)) REDUCE else SHIFT
      }
    new ParseDecision(goldLRN, shiftOrReduceOrPass, getGoldLabel(state))
  }

  def getGoldLabel(state: ParseState): String = {
      val i = state.stackToken(-1)
      if(i > -1 && state.goldHeads(state.stack) == i && !postponeAttachment(i, state.stack, state)) state.goldLabels(state.stack) // right arc
      else if (i > 0 && state.goldHeads(i) == state.stack && !postponeAttachment(state.stack, i, state)) state.goldLabels(i) // left arc
      else ParseDecisionDomain.defaultLabel
  }

  def getGoldLRN(state: ParseState): Int = {
      val j = state.stack
      val i = state.stackToken(-1)
      if (i > 0 && state.goldHeads(i) == j) LEFT
      else if (i >= 0 && state.goldHeads(j) == i && !postponeAttachment(i, j, state)) RIGHT
      else NO
  }

  def postponeAttachment(headIndex: Int, depIndex: Int, state: ParseState): Boolean = {
    if(headIndex < depIndex) {
      for (i <- 1 until state.parseSentenceLength if !state.reducedIds.contains(i))
        if (state.goldHeads(i) == depIndex)
          return true
    }
    false
  }

  /* If the node on top of the stack has a head and can be a transitive head of
     the next input token, false else true */
  def shouldGoldReduce(hasHead: Boolean, state: ParseState): Boolean = {
    if (!hasHead && state.headIndices(state.stack) == -1)
      return false
    for (i <- (state.input + 1) until state.parseSentenceLength)
      if (state.goldHeads(i) == state.stack)
        return false
    true
  }
}