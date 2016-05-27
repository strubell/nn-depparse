package edu.umass.cs.iesl.nndepparse

import cc.factorie.app.nlp.pos.{OntonotesForwardPosTagger, PennPosTag, WSJForwardPosTagger}
import cc.factorie.app.nlp.{DocumentAnnotator, Sentence}
import cc.factorie.app.nlp.load.{AutoLabel, GoldLabel}
import cc.factorie.util.JavaHashSet

class ParseOpts extends cc.factorie.util.DefaultCmdOptions {
  val modelFile = new CmdOption("model", "", "STRING", "Serialized model in HDF5 format")
  val dataFilename = new CmdOption("data-file", "", "STRING", "Filename from which to read test data in CoNLL X one-word-per-line format.")
  val mapsDir = new CmdOption("maps-dir", "", "STRING", "Dir under which to look for existing maps to use; If empty write new maps")
  val lowercase = new CmdOption("lowercase", true, "BOOLEAN", "Whether to lowercase the vocab.")
  val replaceDigits = new CmdOption("replace-digits", "0", "STRING", "Replace digits with the given character (do not replace if empty string).")
  val test = new CmdOption("test", false, "BOOLEAN", "Test mode")
  val reps = new CmdOption("reps", 1, "BOOLEAN", "Number of times to evaluate test set (for timing experiments)")
  val posTagger = new CmdOption("postagger", "", "STRING", "pos tagger to use for timing (empty is none)")
  val testPortion = new CmdOption("test-portion", 1.0, "DOUBLE", "Portion of test sentences to use")
}

object Parse extends App {
  val opts = new ParseOpts
  opts.parse(args)

  val featureFunc = FeatureFunctionsFast.computeChenFeatures _

  // load model parameters
  val model = new FeedForwardNN(opts.modelFile.value, featureFunc, 10000)

  IntMaps.loadMaps(opts.mapsDir.value, true)

  // load documents
  // todo allow loading of plain text documents
  val docs = LoadWSJStanfordDeps.fromFilename(opts.dataFilename.value, loadPos=AutoLabel)
  val allDocSentences = docs.flatMap(_.sentences)
  val docSentences = allDocSentences.take(Math.floor(opts.testPortion.value*allDocSentences.length).toInt)

  println(s"Using ${docSentences.length} test sentences")

  // todo get this from IntMaps
  val punctSet = Set("``", "''", ":", ".", ",")

  if(opts.test.value){
    docSentences.take(1).foreach { sentence =>
        println(s"Projective")
        val state = new ParseState(0, 1, JavaHashSet[Int](), new LightweightParseSentence(sentence, opts.lowercase.value, opts.replaceDigits.value))
        // don't want to try to shift, only arc [for projective parsing of non-projective sentences]
        while (state.input < state.parseSentenceLength || state.stack > 0) {
          if (state.stack < 0) {
            state.stack = state.input
            state.input += 1
          }
          else {
//            val feats = featureFunc(state).map(_.mkString(" ")).mkString("\t")
            val prediction = model.predict(state).split(" ")
            val decision = new ParseDecision(prediction(0).toInt, prediction(1).toInt, IntMaps.intToLabelMap(prediction(2).toInt))
            state.print()
            if (state.stack > -1) println(s"goldHeads(stack) = ${state.goldHeads(state.stack)}")
            val stack_1 = state.stackToken(-1)
            if (stack_1 > -1) println(s"goldHeads(stack-1) = ${state.goldHeads(stack_1)}")
            if (state.input < state.parseSentenceLength) println(s"goldHeads(input) = ${state.goldHeads(state.input)}")
//            println(s"features: $feats")
            println(s"decision: ${decision.readableString}")
            println()
            if(state.input != state.parseSentenceLength || !(decision.shiftOrReduceOrPass == ParserConstants.SHIFT && decision.leftOrRightOrNo == ParserConstants.NO)) {
              ProjectiveArcStandardShiftReduce.transition(state, decision)
            }
            else{
              state.stack = 0
              state.input = state.input + 1
            }
          }
        }
        println("FINAL:")
        state.print()
        println()
    }
  }
  else {
    // parse
    for(i <- 1 to opts.reps.value) {
        val startTime = System.currentTimeMillis()
        val (las, uas, pos, toksPerSec, sentsPerSec) = evaluate(docSentences, punctSet)
        val totalTime = System.currentTimeMillis()-startTime
        println(f"Test LAS: $las%2.2f UAS: $uas%2.2f POS: $pos%2.2f $toksPerSec tokens/sec $sentsPerSec sentences/sec")
    }
  }

  def evaluate(sentences: Seq[Sentence], punctSet: Set[String]): (Double, Double, Double, Double, Double) = {
    var tokenCount = 0
    var sentenceCount = 0
    var nonPunctTokenCount = 0
    var lasCount = 0.0
    var uasCount = 0.0
    var posCount = 0.0
    var totalTime = 0.0
    sentences.foreach{ s =>
      val sentence = new LightweightParseSentence(s, opts.lowercase.value, opts.replaceDigits.value)
      val startTime = System.currentTimeMillis()
      if(opts.posTagger.value != ""){
        WSJForwardPosTagger.process(s)
      }
      val(heads, labels) = ProjectiveArcStandardShiftReduce.parse(sentence, model.predict)
      totalTime += System.currentTimeMillis()-startTime
      var i = 1
      while(i < sentence.length){
        val token = sentence(i)
        if(!punctSet.contains(token.goldPosTagString)){
          nonPunctTokenCount += 1
          if(sentence.goldHeads(i) == heads(i)){
            uasCount += 1
            if(sentence.goldLabels(i) == labels(i)) {
              lasCount += 1
            }
          }
        }
        if(token.goldPosTagString == token.t.attr[PennPosTag].categoryValue)
          posCount += 1
        i += 1
      }
      tokenCount += sentence.length-1
      sentenceCount += 1
    }
    (lasCount/nonPunctTokenCount*100, uasCount/nonPunctTokenCount*100, posCount/tokenCount*100, tokenCount*1000.0/totalTime, sentenceCount*1000.0/totalTime)
  }
}