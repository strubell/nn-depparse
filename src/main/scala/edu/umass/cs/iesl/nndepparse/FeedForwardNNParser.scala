package edu.umass.cs.iesl.nndepparse

import cc.factorie.app.nlp.pos.PosTag
import cc.factorie.app.nlp.{Document, DocumentAnnotator, Sentence, Token}

class FeedForwardNNParser(modelFile: String, mapsDir: String, numToPrecompute: Int) extends DocumentAnnotator {

  val featureFunction = FeatureFunctionsFast.computeChenFeatures _
  val model = new FeedForwardNN(modelFile, featureFunction, numToPrecompute)

  IntMaps.loadMaps(mapsDir, true)

  def prereqAttrs = Seq(classOf[Sentence], classOf[PosTag]) // Sentence also includes Token
  def postAttrs = Seq(classOf[StanfordParseTree])
  override def tokenAnnotationString(token:Token): String = {
    val sentence = token.sentence
    val pt = if (sentence ne null) sentence.attr[StanfordParseTree] else null
    if (pt eq null) "_\t_"
    else (pt.parentIndex(token.positionInSentence)+1).toString+"\t"+pt.label(token.positionInSentence).categoryValue
  }

  def setParse(parseTree: StanfordParseTree, heads: Array[Int], labels: Array[String]) = {
    for(i <- 1 until heads.length){
      val headIndex = heads(i)
      parseTree.setParent(i-1, headIndex-1)
      parseTree.label(i-1).set(StanfordParseTreeLabelDomain.index(labels(i)))(null)
    }
  }

  override def process(document: Document): Document = { document.sentences.foreach(process); document }
  def process(s: Sentence): Sentence = {
    val parseTree = s.attr.getOrElseUpdate(new StanfordParseTree(s))
    val (heads, labels) = ProjectiveArcStandardShiftReduce.parse(new LightweightParseSentence(s), model.predict)
    println(s.tokensString(" "))
    println(heads.drop(1).mkString(" "))
    println(labels.drop(1).mkString(" "))

    setParse(parseTree, heads, labels)
    s
  }
}
