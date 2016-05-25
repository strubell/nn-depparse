package edu.umass.cs.iesl.nndepparse

object ParserConstants {
  val NOTHING = -1

  val ROOT_ID = 0

  val SHIFT  = 1
  val REDUCE = 2
  val PASS   = 3

  val LEFT  = 4
  val RIGHT = 5
  val NO    = 6


  val NULL_STRING = "<NULL>"
  val ROOT_STRING = "<ROOT>"
  val NONE_STRING = "<NONE>"

  val SEP = "|"

  // for debugging purposes
  def apply(i: Int): String = i match {
    case NOTHING => "nothing"

    case SHIFT => "shift"
    case REDUCE => "reduce"
    case PASS => "pass"

    case LEFT => "left"
    case RIGHT => "right"
    case NO => "no"

    case ROOT_ID => "root id"

    case _ => throw new Error(s"Integer value $i is not defined in ParserConstants")
  }

  // todo somewhat concerned about this regex being compiled each time
  def processString(s: String, lowercase: Boolean, replaceDigits: String) = {
    var str = s
    if(replaceDigits != "") str = str.replaceAll("""\d""", replaceDigits)
    if(lowercase) str.toLowerCase else str
  }
}
