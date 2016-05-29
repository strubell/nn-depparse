package edu.umass.cs.iesl.nndepparse

import java.io.{File, PrintWriter}

import cc.factorie.app.nlp.pos.PennPosDomain

class MapToIntsOpts extends cc.factorie.util.DefaultCmdOptions {
  val dataFileFile = new CmdOption("data-file-file", false, "BOOLEAN", "Whether passed data file is a file containing a list of files or a data file itself.")
  val dataFilename = new CmdOption("data-file", "", "STRING", "Filename from which to read training data output by ProcessPTB.")
  val outputDir = new CmdOption("output-dir", "", "STRING", "Dir under which to which to write output maps.")
  val dataOutputFile = new CmdOption("data-output-file", "", "STRING", "File to write the int-mapped data (classlabel -> features).")
  val mapsDir = new CmdOption("maps-dir", "", "STRING", "Dir under which to look for existing maps to use; If empty write new maps")
  val lowercase = new CmdOption("lowercase", true, "BOOLEAN", "Whether to lowercase the vocab.")
  val replaceDigits = new CmdOption("replace-digits", "0", "STRING", "Replace digits with the given character (do not replace if empty string).")
  val feats =  new CmdOption("feats", "chen", "STRING", "Whether loading sprop or chen-style feats (or other...)")
  val loadMaps = new CmdOption("load-maps", true, "BOOLEAN", "Whether to load maps.")

}

object MapFeatsToInts extends App {
  val opts = new MapToIntsOpts
  opts.parse(args)

  val loadMapsFromFile = opts.loadMaps.value
  val mapsDir = if(loadMapsFromFile) opts.mapsDir.value else opts.outputDir.value

  // always load the maps, but maybe don't use them
  IntMaps.loadMaps(mapsDir, false)
  if(!loadMapsFromFile){
    // zero out the non-word maps
    IntMaps.decisionToIntMap.clear()
    IntMaps.labelToIntMap.clear()
    IntMaps.posToIntMap.clear()
  }

  println(s"Loaded size ${IntMaps.wordToIntMap.size} word vocab")

  val loadDecisionMapFromFile = IntMaps.decisionToIntMap.nonEmpty

  val FeatRegex = """([a-z0-9]+)@[sbl]-?[0-9h]=(.+)""".r

  /* Lua is 1-indexed; start counters at 1 */
  IntMaps.wordToIntMap("<OOV>") = 1
  IntMaps.wordToIntMap(ParserConstants.NULL_STRING) = 2
  IntMaps.wordToIntMap(ParserConstants.ROOT_STRING) = 3

  IntMaps.posToIntMap(ParserConstants.NULL_STRING) = 1
  IntMaps.posToIntMap(ParserConstants.ROOT_STRING) = 2

  IntMaps.labelToIntMap(ParserConstants.NULL_STRING) = 1

  var wordCounter = IntMaps.wordToIntMap.size + 1
  var posCounter = IntMaps.posToIntMap.size + 1
  var labelCounter = IntMaps.labelToIntMap.size + 1
  var decisionCounter = IntMaps.decisionToIntMap.size + 1

  val dataFnames = if(opts.dataFileFile.value) io.Source.fromFile(opts.dataFilename.value).getLines().toSeq else Seq(opts.dataFilename.value)
  val dataOutput = new PrintWriter(opts.outputDir.value + "/" + opts.dataOutputFile.value, "utf-8")

  dataFnames.foreach{fname =>
    val inputSrc = io.Source.fromFile(fname, "utf-8")
    var i = 1

    inputSrc.getLines().foreach { line =>
      if (line.trim != "") {
        val splitLine = line.split("\t")
        val decision = splitLine(0)
        // handle collapsed
        val splitCollapsedDecision = decision.split("/")
        val intmappedCollapsedDecision = splitCollapsedDecision.map{decision =>
          val splitDecision = decision.trim.split(" ")

          val decisionLabel = splitDecision(2).toInt
          val decisionLabelString = StanfordParseTreeLabelDomain.category(decisionLabel)
          if (!IntMaps.labelToIntMap.contains(decisionLabelString)) {
            IntMaps.labelToIntMap(decisionLabelString) = labelCounter
            labelCounter += 1
          }
          val intmappedDecisionLabel = IntMaps.labelToIntMap(decisionLabelString)
          if(splitDecision.length > 3) {
            val posString = PennPosDomain.category(splitDecision(3).toInt)
            if (!IntMaps.posToIntMap.contains(posString)) {
              IntMaps.posToIntMap(posString) = posCounter
              posCounter += 1
            }
            val intmappedPosLabel = IntMaps.posToIntMap(posString)
            s"${splitDecision(0)} ${splitDecision(1)} $intmappedDecisionLabel $intmappedPosLabel"
          }
          else s"${splitDecision(0)} ${splitDecision(1)} $intmappedDecisionLabel"
        }.mkString("/")


        if (!loadDecisionMapFromFile && !IntMaps.decisionToIntMap.contains(intmappedCollapsedDecision)) {
          IntMaps.decisionToIntMap(intmappedCollapsedDecision) = decisionCounter
          decisionCounter += 1
        }

        // only write if decision in train
        if(IntMaps.decisionToIntMap.contains(intmappedCollapsedDecision)) {
          dataOutput.print(IntMaps.decisionToIntMap(intmappedCollapsedDecision) + "\t")

          assert(splitLine.size == 4, splitLine.mkString(" "))
          val wordFeats = splitLine(1).split(" ")
          val posFeats = splitLine(2).split(" ")
          val labelFeats = splitLine(3).split(" ")

          val wordFeatInts = wordFeats.map { feat =>
            val FeatRegex(featureType, featureVal) = feat
            IntMaps.wordToIntMap.getOrElse(featureVal, 1)
          }

          val posFeatInts = posFeats.map { feat =>
            val FeatRegex(featureType, featureVal) = feat
            if (!loadMapsFromFile && !IntMaps.posToIntMap.contains(featureVal)) {
              IntMaps.posToIntMap(featureVal) = posCounter
              posCounter += 1
            }
            if (!IntMaps.posToIntMap.contains(featureVal)) println(s"POS label $featureVal not seen in training. Feature: ${feat}")
            IntMaps.posToIntMap.getOrElse(featureVal, 1)
          }

          val labelFeatInts = labelFeats.map { feat =>
            val FeatRegex(featureType, featureVal) = feat
            if (!loadMapsFromFile && !IntMaps.labelToIntMap.contains(featureVal)) {
              IntMaps.labelToIntMap(featureVal) = labelCounter
              labelCounter += 1
            }
            if (!IntMaps.labelToIntMap.contains(featureVal)) println(s"Label $featureVal not seen in training")
            IntMaps.labelToIntMap.getOrElse(featureVal, 1)
          }
          dataOutput.print(wordFeatInts.mkString(" ") + "\t")
          dataOutput.print(posFeatInts.mkString(" ") + "\t")
          dataOutput.print(labelFeatInts.mkString(" "))
          dataOutput.println()
        }
      }
      else {dataOutput.println(); i += 1}
    }
    inputSrc.close()
  }
  dataOutput.close()

  val mapFilenames = new MapFilenames(mapsDir)

  /* Write label maps regardless */
  if(!loadDecisionMapFromFile) {
    val decisionToIntOutput = new PrintWriter(mapFilenames.decisionToIntMap, "utf-8")
    println("writing decision map length " + IntMaps.decisionToIntMap.size)
    IntMaps.decisionToIntMap.foreach { case (k, v) => decisionToIntOutput.println(k + "\t" + v) }
    decisionToIntOutput.close()
  }

  /* Write other maps if none were loaded */
  if(!loadMapsFromFile){

    println("writing word map length " + IntMaps.wordToIntMap.size)
    println("writing pos map length " + IntMaps.posToIntMap.size)
    println("writing label map length " + IntMaps.labelToIntMap.size)

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
