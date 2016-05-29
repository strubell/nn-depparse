package edu.umass.cs.iesl.nndepparse

import cc.factorie.app.nlp._
import collection.mutable.ArrayBuffer
import cc.factorie.util.Cubbie
import cc.factorie.variable.{LabeledCategoricalVariable, EnumDomain}

// Representation for a dependency parse

// Stanford Dependencies
object StanfordParseTreeLabelDomain extends EnumDomain { // this is for Stanford dependencies
val root, dep, aux, auxpass, cop, arg, agent, comp, acomp, ccomp, xcomp, obj, dobj, iobj, pobj, subj, nsubj,
nsubjpass, csubj, csubjpass, cc, conj, expl, mod, amod, appos, advcl, det, predet, preconj, vmod, mwe, mark,
advmod, neg, rcmod, quantmod, nn, npadvmod, tmod, num, number, prep, poss, possessive, prt, parataxis, punct,
ref, sdep, xsubj, pcomp, discourse = Value
  index(ParserConstants.NULL_STRING) // necessary for empty categories
  freeze()
  def defaultCategory = "nn"
  def defaultIndex = index(defaultCategory)
}

// TODO I think this should instead be "ParseEdgeLabels extends LabeledCategoricalSeqVariable". -akm
class ParseTreeLabel(val tree:StanfordParseTree, targetValue:String = StanfordParseTreeLabelDomain.defaultCategory) extends LabeledCategoricalVariable(targetValue) { def domain = StanfordParseTreeLabelDomain }

object StanfordParseTree {
  val rootIndex = -1
  val noIndex = -2
}

// TODO This initialization is really inefficient.  Fix it. -akm
class StanfordParseTree(val sentence:Sentence, theTargetParents:Array[Int], theTargetLabels:Array[Int]) {
  def this(sentence:Sentence) = this(sentence, Array.fill[Int](sentence.length)(StanfordParseTree.noIndex), Array.fill(sentence.length)(StanfordParseTreeLabelDomain.defaultIndex)) // Note: this puts in dummy target data which may be confusing
  def this(sentence:Sentence, theTargetParents:Seq[Int], theTargetLabels:Seq[String]) = this(sentence, theTargetParents.toArray, theTargetLabels.map(c => StanfordParseTreeLabelDomain.index(c)).toArray)
  def check(parents:Array[Int]): Unit = {
    val l = parents.length; var i = 0; while (i < l) {
      require(parents(i) < l)
      i += 1
    }
  }
  check(theTargetParents)
  val _labels = theTargetLabels.map(s => {assert(StanfordParseTreeLabelDomain.category(s) != -1, s"dep ${s} not in domain."); new ParseTreeLabel(this, StanfordParseTreeLabelDomain.category(s))}).toArray
  val _parents = { val p = new Array[Int](theTargetParents.length); System.arraycopy(theTargetParents, 0, p, 0, p.length); p }
  val _targetParents = theTargetParents
  val _targetLabels = theTargetLabels
  def labels: Array[ParseTreeLabel] = _labels
  def parents: Array[Int] = _parents
  def targetParents: Array[Int] = _targetParents
  def setParentsToTarget(): Unit = System.arraycopy(_targetParents, 0, _parents, 0, _parents.length)
  def numParentsCorrect: Int = { var result = 0; for (i <- 0 until _parents.length) if (_parents(i) == _targetParents(i)) result += 1; result }
  def parentsAccuracy: Double = numParentsCorrect.toDouble / _parents.length
  def numLabelsCorrect: Int = {var result = 0; for (i <- 0 until _labels.length) if (_labels(i).valueIsTarget) result += 1; result }
  def labelsAccuracy: Double = numLabelsCorrect.toDouble / _labels.length
  /** Returns the position in the sentence of the root token. */
  def rootChildIndex: Int = firstChild(-1)
  /** Return the token at the root of the parse tree.  The parent of this token is null.  The parentIndex of this position is -1. */
  def rootChild: Token = sentence.tokens(rootChildIndex)
  /** Make the argument the root of the tree.  This method does not prevent their being two roots. */
  def setRootChild(token:Token): Unit = setParent(token.position - sentence.start, -1)
  /** Returns the sentence position of the parent of the token at position childIndex */
  def parentIndex(childIndex:Int): Int = if (childIndex == StanfordParseTree.rootIndex) StanfordParseTree.noIndex else _parents(childIndex)
  def targetParentIndex(childIndex:Int): Int = if (childIndex == StanfordParseTree.rootIndex) StanfordParseTree.noIndex else _targetParents(childIndex)
  /** Returns the parent token of the token at position childIndex (or null if the token at childIndex is the root) */
  def parent(childIndex:Int): Token = {
    val idx = _parents(childIndex)
    if (idx == -1) null // -1 is rootIndex
    else sentence.tokens(idx)
  }
  /** Returns the parent token of the given token */
  def parent(token:Token): Token = { require(token.sentence eq sentence); parent(token.position - sentence.start) }
  /** Set the parent of the token at position 'child' to be at position 'parentIndex'.  A parentIndex of -1 indicates the root.  */
  def setParent(childIndex:Int, parentIndex:Int): Unit = _parents(childIndex) = parentIndex
  def setTargetParent(childIndex:Int, parentIndex:Int): Unit = _targetParents(childIndex) = parentIndex
  /** Set the parent of the token 'child' to be 'parent'. */
  def setParent(child:Token, parent:Token): Unit = {
    require(child.sentence eq sentence)
    if (parent eq null) {
      _parents(child.position - sentence.start) = -1
    } else {
      require(parent.sentence eq sentence)
      _parents(child.position - sentence.start) = parent.position - sentence.start
    }
  }

  //TODO: all of the  following methods are inefficient if the parse tree is fixed, and various things
  //can be precomputed.

  /** Return the sentence index of the first token whose parent is 'parentIndex' */
  private def firstChild(parentIndex:Int): Int = {
    var i = 0
    while ( i < _parents.length) {
      if (_parents(i) == parentIndex) return i
      i += 1
    }
    -1
  }


  /** Return a list of tokens who are the children of the token at sentence position 'parentIndex' */
  def children(parentIndex:Int): Seq[Token] = {
    getChildrenIndices(parentIndex).map(i => sentence.tokens(i))
  }

  def getChildrenIndices(parentIndex:Int, filter : Int => Boolean = {x => false}): Seq[Int] = {
    val result = new ArrayBuffer[Int]
    var i = 0
    while (i < _parents.length) {
      if (_parents(i) == parentIndex) result += i
      i += 1
    }
    result.sorted.takeWhile( i => !filter(i))
  }

  def subtree(parentIndex:Int): Seq[Token] = {
    getSubtreeInds(parentIndex).map(sentence.tokens(_))
  }

  def getSubtreeInds(parentIndex: Int, filter : Int => Boolean = {x => false}): Seq[Int] = {
    val result = new ArrayBuffer[Int]()
    result += parentIndex
    result ++= getChildrenIndices(parentIndex, filter).flatMap(getSubtreeInds(_)).distinct
    result
  }

  def leftChildren(parentIndex:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < parentIndex) {
      if (_parents(i) == parentIndex) result += sentence.tokens(i)
      i += 1
    }
    result
  }
  def rightChildren(parentIndex:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = parentIndex+1
    while (i < _parents.length) {
      if (_parents(i) == parentIndex) result += sentence.tokens(i)
      i += 1
    }
    result
  }
  /** Return a list of tokens who are the children of parentToken */
  //def children(parentToken:Token): Seq[Token] = children(parentToken.position - sentence.start)
  /** Return a list of tokens who are the children of the token at sentence position 'parentIndex' and who also have the indicated label value. */
  def childrenLabeled(index:Int, labelIntValue:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < _parents.length) {
      if (_parents(i) == index && _labels(i).intValue == labelIntValue) result += sentence.tokens(i)
      i += 1
    }
    result
  }
  def leftChildrenLabeled(parentIndex:Int, labelIntValue:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < parentIndex) {
      if (_parents(i) == parentIndex && _labels(i).intValue == labelIntValue) result += sentence.tokens(i)
      i += 1
    }
    result
  }
  def rightChildrenLabeled(parentIndex:Int, labelIntValue:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = parentIndex+1
    while (i < _parents.length) {
      if (_parents(i) == parentIndex && _labels(i).intValue == labelIntValue) result += sentence.tokens(i)
      i += 1
    }
    result
  }
  //def childrenOfLabel(token:Token, labelIntValue:Int): Seq[Token] = childrenOfLabel(token.position - sentence.start, labelIntValue)
  //def childrenLabeled(index:Int, labelValue:DiscreteValue): Seq[Token] = childrenLabeled(index, labelValue.intValue)
  //def childrenOfLabel(token:Token, labelValue:DiscreteValue): Seq[Token] = childrenOfLabel(token.position - sentence.start, labelValue.intValue)
  /** Return the label on the edge from the child at sentence position 'index' to its parent. */
  def label(index:Int): ParseTreeLabel = _labels(index)
  def copy: StanfordParseTree = {
    val newTree = new StanfordParseTree(sentence, targetParents, labels.map(_.target.categoryValue))
    for (i <- 0 until sentence.length) {
      newTree._parents(i) = this._parents(i)
      newTree._labels(i).set(this._labels(i).intValue)(null)
    }
    newTree
  }
  /** Return the label on the edge from 'childToken' to its parent. */
  //def label(childToken:Token): ParseTreeLabel = { require(childToken.sentence eq sentence); label(childToken.position - sentence.start) }
  override def toString: String = {
    val tokenStrings = {
      if (sentence.tokens.forall(_.posTag ne null))
        sentence.tokens.map(t => t.string + "/" + t.posTag.categoryValue)
      else
        sentence.tokens.map(_.string)
    }
    val labelStrings = _labels.map(_.value.toString())
    val buff = new StringBuffer()
    for (i <- 0 until sentence.length)
      buff.append(i + " " + _parents(i) + " " + tokenStrings(i) + " " + labelStrings(i) + "\n")
    buff.toString
  }

  def toStringTex:String = {
    def texEdges(idx:Int, builder:StringBuilder):StringBuilder = this.children(idx) match {
      case empty if empty.isEmpty => builder
      case children => children.foreach { token =>
        val childIdx = token.positionInSentence
        val parentIdx = token.parseParentIndex
        val label = token.parseLabel.categoryValue
        builder.append("  \\depedge{%s}{%s}{%s}".format(parentIdx + 1, childIdx + 1, label)).append("\n") // latex uses 1-indexing
        texEdges(childIdx, builder)
      }
        builder
    }
    val sentenceString = this.sentence.tokens.map(_.string).mkString(""" \& """) + """\\"""

    val rootId = this.rootChildIndex
    val rootLabel = this.label(rootId).categoryValue // should always be 'root'
    val rootString = "  \\deproot{%s}{%s}".format(rootId, rootLabel)

    val sb = new StringBuilder
    sb.append("""\begin{dependency}""").append("\n")
    sb.append("""  \begin{deptext}""").append("\n")
    sb.append(sentenceString).append("\n")
    sb.append("""  \end{deptext}""").append("\n")
    sb.append(rootString).append("\n")
    texEdges(rootId, sb)
    sb.append("""\end{dependency}""").append("\n")

    sb.toString()
  }

}

// Example usages:
// token.sentence.attr[ParseTree].parent(token)
// sentence.attr[ParseTree].children(token)
// sentence.attr[ParseTree].setParent(token, parentToken)
// sentence.attr[ParseTree].label(token)
// sentence.attr[ParseTree].label(token).set("SUBJ")

// Methods also created in Token supporting:
// token.parseParent
// token.setParseParent(parentToken)
// token.parseChildren
// token.parseLabel
// token.leftChildren

class ParseTreeCubbie extends Cubbie {
  val parents = IntListSlot("parents")
  val labels = StringListSlot("labels")
  def newParseTree(s:Sentence): StanfordParseTree = new StanfordParseTree(s) // This will be abstract when ParseTree domain is unfixed
  def storeParseTree(pt:StanfordParseTree): this.type = {
    parents := pt.parents
    labels := pt.labels.map(_.categoryValue)
    this
  }
  def fetchParseTree(s:Sentence): StanfordParseTree = {
    val pt = newParseTree(s)
    for (i <- 0 until s.length) {
      pt.setParent(i, parents.value(i))
      pt.label(i).setCategory(labels.value(i))(null)
    }
    pt
  }
}