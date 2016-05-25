package edu.umass.cs.iesl.nndepparse

import cc.factorie.app.nlp.{Sentence, Token}
import cc.factorie.app.nlp.pos.PosTag

class LightweightParseToken(val t: Token, lowercase: Boolean = false, replaceDigits: String = ""){
  lazy val rawString = t.string
  lazy val string = ParserConstants.processString(t.string, lowercase, replaceDigits)
  lazy val posTag = t.attr[PosTag]
  lazy val goldPosTagString = t.attr[GoldPos].categoryValue
  lazy val posTagString = if(posTag ne null) posTag.categoryValue else string
}

object RootToken extends LightweightParseToken(null.asInstanceOf[Token]){
  override lazy val string = ParserConstants.ROOT_STRING
  override lazy val posTagString = ParserConstants.ROOT_STRING
}
object NullToken extends LightweightParseToken(null.asInstanceOf[Token]){
  override lazy val string = ParserConstants.NULL_STRING
  override lazy val posTagString = ParserConstants.NULL_STRING
}

class LightweightParseSentence(s: Sentence, lowercase: Boolean = false, replaceDigits: String = ""){
  val length: Int = s.length + 1
  val _tokens: Array[LightweightParseToken] = new Array[LightweightParseToken](length-1)
  var i = 0; while(i < length-1) { _tokens(i) = new LightweightParseToken(s(i), lowercase, replaceDigits); i += 1 }
  val parse = s.attr[ParseTree]
  val goldHeads = Seq(-1) ++ parse._targetParents.map(_ + 1)
  val goldLabels = Seq("<ROOT-ROOT>") ++ parse._labels.map(_.target.categoryValue)

  // we are working with the original sentence, with an additional
  // ROOT token that comes at index 0, moving all other indices up by 1:
  // idx < 0 -> NULL_TOKEN
  // idx = 0 -> ROOT_TOKEN
  // 0 < idx < sentence.length+1 -> sentence(idx-1)
  // idx > sentence.length -> NULL_TOKEN
  def apply(idx: Int) = idx match {
//    case 0 => RootToken
    case i if (i > 0 && i < length) => _tokens(i-1)
    case _ => NullToken
  }
}