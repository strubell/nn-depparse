package edu.umass.cs.iesl.nndepparse

import java.io.PrintWriter

import cc.factorie.app.nlp.load.{GoldLabel, AutoLabel}
import cc.factorie.util.{Threading, JavaHashSet}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

class ProcessPTBOpts extends cc.factorie.util.DefaultCmdOptions {
  val dataFileFile = new CmdOption("data-file-file", false, "BOOLEAN", "Whether passed data file is a file containing a list of files or a data file itself.")
  val dataFilename = new CmdOption("data-file", "", "STRING", "Filename from which to read training data in CoNLL X one-word-per-line format.")
  val outputDir = new CmdOption("output-dir", "", "STRING", "Filename to which to write output.")
  val outputAppend = new CmdOption("output-append", "decisions", "STRING", "string to append to output filenames")
  val lowercase = new CmdOption("lowercase", true, "BOOLEAN", "Whether to lowercase the vocab.")
  val replaceDigits = new CmdOption("replace-digits", "0", "STRING", "Replace digits with the given character (do not replace if empty string).")
  val pos = new CmdOption("pos", "auto", "STRING", "Which pos tags to use: gold, auto")
  val embeddingsFile = new CmdOption("embeddings", "", "STRING", "File containing word embeddings for filtering vocab")
  val mapsDir = new CmdOption("maps-dir", "", "STRING", "Dir under which to look for existing maps to use; If empty write new maps")
  val writeMap = new CmdOption("write-map", false, "BOOLEAN", "Whether to write word map")
  val parallel = new CmdOption("parallel", 1, "INT", "Parallelize to given number of cores")
  val filterProjective = new CmdOption("filter-projective", true, "BOOLEAN", "Whether to filter non-projective trees out of data")
  val test = new CmdOption("test", false, "BOOLEAN", "Whether to print verbose testing output")
}

object ProcessPTB extends App {
  val opts = new ProcessPTBOpts
  opts.parse(args)
  println(s"Processing ${opts.dataFilename.value}")

  val docFilenames =
    if(opts.dataFileFile.value)
      io.Source.fromFile(opts.dataFilename.value).getLines().toSeq
    else Seq(opts.dataFilename.value)

  def outputString(inputFname: String) = s"${opts.outputDir.value}/${inputFname.split("/").last}.${opts.outputAppend.value}"

  val outputFilenames =
    if(opts.dataFileFile.value)
    io.Source.fromFile(opts.dataFilename.value).getLines().map(outputString).toSeq
  else Seq(outputString(opts.dataFilename.value))

  val featureFunc = FeatureFunctions.computeChenFeatures _

  val parser = ProjectiveArcStandardShiftReduce

  val wordVocab = collection.parallel.mutable.ParHashSet[String]()

  val embeddedWords = io.Source.fromFile(opts.embeddingsFile.value, "utf-8").getLines().drop(1).map(line => line.split(" ").head).toSet

  def processDoc(fname: String, idx: Int): (Double, Double) = {
    val writer = new PrintWriter(outputFilenames(idx), "utf-8")

    val docs = opts.pos.value match {
      case "gold" => LoadWSJStanfordDeps.fromFilename(fname, loadPos=GoldLabel)
      case "auto" => LoadWSJStanfordDeps.fromFilename(fname, loadPos=AutoLabel)
      case p => throw new Error(s"POS tag type `$p' not defined")
    }

    val docWordCounts = collection.mutable.HashMap[String,Int]()

    var nonProjCount = 0.0
    var sentencesCount = 0.0
    docs.foreach{ doc =>
      doc.sentences.foreach{s =>
      val sentence = new LightweightParseSentence(s, opts.lowercase.value, opts.replaceDigits.value)
        sentence._tokens.foreach{token => docWordCounts.getOrElseUpdate(token.string, 1)}
        if (!opts.filterProjective.value || isProjective(sentence)) {
          val sentenceFeats = parser.getGoldDecisionFeatures(sentence, featureFunc)
          sentenceFeats.foreach {
            writer.println
          }
          writer.println()
        }
        else nonProjCount += 1.0
        sentencesCount += 1.0
      }
      writer.close()
    }

    // gross hack because we want to keep all the words in wsj
    if(outputFilenames(idx).contains("wsj")) wordVocab ++= docWordCounts.keys
    else {
      wordVocab ++= docWordCounts.filterKeys(k => embeddedWords.contains(k)).keys
      println(s"Filtered ${docWordCounts.filterKeys(k => !embeddedWords.contains(k)).size} tokens")
    }

    (nonProjCount, sentencesCount)
  }


  if(opts.test.value){
    val doc = opts.pos.value match {
      case "gold" => LoadWSJStanfordDeps.fromFilename(docFilenames.head, loadPos=GoldLabel)
      case "auto" => LoadWSJStanfordDeps.fromFilename(docFilenames.head, loadPos=AutoLabel)
      case p => throw new Error(s"POS tag type `$p' not defined")
    }
    val docSentences = doc.flatMap{d => d.sentences.map{sentence => new LightweightParseSentence(sentence, opts.lowercase.value, opts.replaceDigits.value)}}
    docSentences.foreach { sentence =>
      if (!opts.filterProjective.value || isProjective(sentence)) {
        println(s"Projective")
        val state = new ParseState(0, 1, JavaHashSet[Int](), sentence)
        // don't want to try to shift, only arc [for projective parsing of non-projective sentences]
        while (state.input < state.parseSentenceLength || state.stack > 0) {
          if (state.stack < 0) {
            state.stack = state.input
            state.input += 1
          }
          else {
            val feats = featureFunc(state).map(_.mkString(" ")).mkString("\t")
            val decision = parser.getGoldDecision(state)
            state.print()
            if (state.stack > -1) println(s"goldHeads(stack) = ${state.goldHeads(state.stack)}")
            val stack_1 = state.stackToken(-1)
            if (stack_1 > -1) println(s"goldHeads(stack-1) = ${state.goldHeads(stack_1)}")
            if (state.input < state.parseSentenceLength) println(s"goldHeads(input) = ${state.goldHeads(state.input)}")
            println(s"features: $feats")
            println(s"decision: ${decision.readableString}")
            println()
            if(state.input != state.parseSentenceLength || !(decision.shiftOrReduceOrPass == ParserConstants.SHIFT && decision.leftOrRightOrNo == ParserConstants.NO))
              parser.transitionOnGoldLabel(state)
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
      else println("Warning: encountered non-projective sentence")
    }
  }
  else {
    val sentenceCounts =
      if(opts.parallel.value > 1)
        Threading.parMap(docFilenames.zipWithIndex, opts.parallel.value){case(doc, docIdx) => processDoc(doc, docIdx)}
      else
        docFilenames.zipWithIndex.map{case(doc, docIdx) => processDoc(doc, docIdx)}
    val numSentences = sentenceCounts.map(_._2).sum
    val nonProjCount = sentenceCounts.map(_._1).sum
    val projectivePercent = (1.0-(nonProjCount/numSentences))*100
    println(f"Processed $numSentences sentences ($projectivePercent%2.2f projective)")

    // also write fnames file if multiple files processed
    if(opts.dataFileFile.value) {
      val writer = new PrintWriter(outputString(opts.dataFilename.value))
      outputFilenames.foreach { fname => writer.println(fname) }
      writer.close()
    }

    // write word vocab
    if(opts.writeMap.value) {
      println(s"Writing vocab of ${wordVocab.size} words")
      val mapFilenames = new MapFilenames(opts.mapsDir.value)
      val writer = new PrintWriter(mapFilenames.wordToIntMap)
      // add 4 because 1, 2 and 3 are taken by OOV, NULL and ROOT tokens
      wordVocab.zipWithIndex.foreach { case (w, i) => writer.println(s"$w\t${i + 4}") }
      writer.close()
    }
  }

  @tailrec def isDescendantOf(firstIndex: Int, secondIndex: Int, sentence: LightweightParseSentence): Boolean = {
    val firstHeadIndex = sentence.goldHeads(firstIndex)
    if (firstHeadIndex == -1) false // firstIndex has no head, so it can't be a descendant
    else if (firstHeadIndex == secondIndex) true
    else isDescendantOf(firstHeadIndex, secondIndex, sentence)
  }

  /* A sentence is projective if all arcs are projective, and and arc from head w_i
     to dependent w_j is projective if w_i is an ancestor of each word w_k between
     w_i and w_j */
  def isProjective(sentence: LightweightParseSentence): Boolean = {
    sentence.goldHeads.zipWithIndex.drop(1).foreach { case (head, dep) =>
      if (head != 0) {
        val (start, end) = if(head < dep) (head, dep) else (dep, head)
        (start + 1 until end).foreach { idx =>
          if (!isDescendantOf(idx, head, sentence)) {
//            println(s"$idx is not descendant of $head (checking ${start+1} until $end for arc $head->$dep)")
            return false
          }
        }
      }
    }
    true
  }
}
