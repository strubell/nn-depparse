package edu.umass.cs.iesl.nndepparse

import java.io.PrintWriter

import cc.factorie.app.nlp.load.AutoLabel

object MapSentsToInts extends App {
  val opts = new MapToIntsOpts
  opts.parse(args)

  val loadMapsFromFile = opts.loadMaps.value
  val mapsDir = opts.mapsDir.value


  IntMaps.loadMaps(mapsDir, false)
  val mapFilenames = new MapFilenames(mapsDir)

  if(!loadMapsFromFile){
    // zero out the non-word maps
    IntMaps.decisionToIntMap.clear()
//    IntMaps.labelToIntMap.clear()
    IntMaps.posToIntMap.clear()
  }

  println(s"Loaded size ${IntMaps.wordToIntMap.size} word vocab")

  // punctuation to ignore during evaluation
  val ignorePunct = Set("``", "''", ":", ".", ",")

  val punctSet = collection.mutable.HashSet[Int]()


  /* Lua is 1-indexed; start counters at 1 */
  IntMaps.wordToIntMap("<OOV>") = 1
  IntMaps.wordToIntMap(ParserConstants.NULL_STRING) = 2
  IntMaps.wordToIntMap(ParserConstants.ROOT_STRING) = 3

  // these have no OOV
  IntMaps.posToIntMap(ParserConstants.NULL_STRING) = 1
  IntMaps.posToIntMap(ParserConstants.ROOT_STRING) = 2

  IntMaps.labelToIntMap(ParserConstants.NULL_STRING) = 1

  var wordCounter = IntMaps.wordToIntMap.size + 1
  var posCounter = IntMaps.posToIntMap.size + 1
  var labelCounter = IntMaps.labelToIntMap.size + 1

  println(opts.dataFileFile.value)

  val dataFnames = if(opts.dataFileFile.value) io.Source.fromFile(opts.dataFilename.value).getLines().toSeq else Seq(opts.dataFilename.value)
  val dataOutput = new PrintWriter(opts.outputDir.value + "/" + opts.dataOutputFile.value, "utf-8")

  dataFnames.foreach { fname =>
    val docs = LoadWSJStanfordDeps.fromFilename(fname, loadPos = AutoLabel)
    assert(docs.length == 1)

    val docSentences = docs.flatMap(_.sentences)

    docSentences.foreach { sentence =>
      val parse = sentence.attr[ParseTree]
      if (parse == null) println(s"Parse null in sentence: ${sentence.tokensString(" ")}")
      sentence.tokens.foreach { token =>
        val word = ParserConstants.processString(token.string, opts.lowercase.value, opts.replaceDigits.value)
        val pos = token.posTag.categoryValue
        val goldPos = token.attr[GoldPos].categoryValue

        val label = parse.label(token.positionInSentence).categoryValue
        val head = parse.parentIndex(token.positionInSentence)

        if (!loadMapsFromFile && !IntMaps.wordToIntMap.contains(word)) {
          IntMaps.wordToIntMap(word) = wordCounter
          wordCounter += 1
        }
        val wordInt = IntMaps.wordToIntMap.getOrElse(word, 1)

        if (!IntMaps.labelToIntMap.contains(label)) {
          IntMaps.labelToIntMap(label) = labelCounter
          labelCounter += 1
        }
        val labelInt = IntMaps.labelToIntMap(label)

        if (!loadMapsFromFile && !IntMaps.posToIntMap.contains(pos)) {
          IntMaps.posToIntMap(pos) = posCounter
          posCounter += 1
        }
        val posInt = IntMaps.posToIntMap(pos)

        if (!loadMapsFromFile && !IntMaps.posToIntMap.contains(goldPos)) {
          IntMaps.posToIntMap(goldPos) = posCounter
          posCounter += 1
        }
        val goldPosInt = IntMaps.posToIntMap(goldPos)
        if (!punctSet.contains(goldPosInt) && ignorePunct.contains(goldPos)) punctSet += goldPosInt

        dataOutput.println(s"$wordInt\t$posInt\t$labelInt\t$head\t$goldPosInt")
      }
      dataOutput.println()
    }
  }
  dataOutput.close()

  val punctOutput = new PrintWriter(mapFilenames.punctSet, "utf-8")
  punctSet.foreach{p => punctOutput.println(p)}
  punctOutput.close()

  /* Write maps if none were loaded */
  if(!loadMapsFromFile){
    val wordOutput = new PrintWriter(mapFilenames.wordToIntMap, "utf-8")
    val posOutput = new PrintWriter(mapFilenames.posToIntMap, "utf-8")
    val labelOutput = new PrintWriter(mapFilenames.labelToIntMap, "utf-8")

    IntMaps.wordToIntMap.foreach{case(k,v) => wordOutput.println(k + "\t" + v)}
    IntMaps.posToIntMap.foreach{case(k,v) => posOutput.println(k + "\t" + v)}
    IntMaps.labelToIntMap.foreach{case(k,v) => labelOutput.println(k + "\t" + v)}

    wordOutput.close()
    posOutput.close()
    labelOutput.close()
  }
}
