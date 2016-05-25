package edu.umass.cs.iesl.nndepparse

import java.io.File

import cc.factorie.util.JavaHashMap

class MapFilenames(dir: String){
  val wordToIntMap = dir + "/" + "word2int"
  val posToIntMap = dir + "/" + "pos2int"
  val labelToIntMap = dir + "/" + "label2int"
  val decisionToIntMap = dir + "/" + "decision2int"
  val punctSet = dir + "/" + "punct"
}

object IntMaps {
  val wordToIntMap = JavaHashMap[String,Int]()
  val posToIntMap = JavaHashMap[String,Int]()
  val labelToIntMap = JavaHashMap[String,Int]()
  val decisionToIntMap = JavaHashMap[String,Int]()

  val intToPosMap = JavaHashMap[Int,String]()
  val intToLabelMap = JavaHashMap[Int,String]()
  val intToDecisionMap = JavaHashMap[Int,String]()

  def loadMaps(mapsDir: String, reversed: Boolean) = {
    println(s"Loading maps from $mapsDir")
    val filenames = new MapFilenames(mapsDir)
    loadMap(filenames.wordToIntMap, wordToIntMap)
    loadMap(filenames.posToIntMap, posToIntMap)
    loadMap(filenames.labelToIntMap, labelToIntMap)
    loadMap(filenames.decisionToIntMap, decisionToIntMap)

    if(reversed){
      loadReverseMap(filenames.posToIntMap, intToPosMap)
      loadReverseMap(filenames.labelToIntMap, intToLabelMap)
      loadReverseMap(filenames.decisionToIntMap, intToDecisionMap)
    }
  }

  def loadMap(filename: String): collection.mutable.Map[String,Int] = loadMap(filename, JavaHashMap[String,Int]())
  def loadMap(filename: String, map: collection.mutable.Map[String,Int]): collection.mutable.Map[String,Int] = {
    // only load if the file actually exists
    if(new File(filename).exists()) {
      val src = io.Source.fromFile(filename)
      src.getLines().foreach { line =>
        val splitLine = line.split("\t")
        assert(splitLine.length == 2)
        map(splitLine(0)) = splitLine(1).toInt
      }
      src.close()
    }
    else{
      println(s"Warning: Unable to load `$filename'; File does not exist.")
    }
    map
  }

  def loadReverseMap(filename: String): collection.mutable.Map[Int,String] = loadReverseMap(filename, JavaHashMap[Int,String]())
  def loadReverseMap(filename: String, map: collection.mutable.Map[Int,String]): collection.mutable.Map[Int,String] = {
    // only load if the file actually exists
    if(new File(filename).exists()) {
      val src = io.Source.fromFile(filename)
      src.getLines().foreach { line =>
        val splitLine = line.split("\t")
        assert(splitLine.length == 2)
        map(splitLine(1).toInt) = splitLine(0)
      }
      src.close()
    }
    else{
      println(s"Warning: Unable to load `$filename'; File does not exist.")
    }
    map
  }

}
