package edu.umass.cs.iesl.nndepparse

import cc.factorie.app.nlp.{Document, DocumentAnnotationPipeline}
import cc.factorie.app.nlp.pos.OntonotesForwardPosTagger
import cc.factorie.app.nlp.segment.{DeterministicNormalizingTokenizer, DeterministicSentenceSegmenter}

object FactoriePipelineExample extends App {

  val opts = new ParseOpts
  opts.parse(args)

  val sentence = "The student studied very hard for her exams."
  val doc = new Document(sentence)

  val FeedForwardNNParser = new FeedForwardNNParser(opts.modelFile.value, opts.mapsDir.value, opts.numToPrecompute.value)

  val annotators = Seq(DeterministicNormalizingTokenizer, DeterministicSentenceSegmenter, OntonotesForwardPosTagger, FeedForwardNNParser)

  val pipeline = new DocumentAnnotationPipeline(annotators)

  pipeline.process(doc)

  println(s"sentences: ${doc.sentenceCount} tokens: ${doc.tokenCount}")

  println(doc.owplString(annotators.map(p => p.tokenAnnotationString(_))))
}
