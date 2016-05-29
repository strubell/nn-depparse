package edu.umass.cs.iesl.nndepparse

import cc.factorie.app.nlp._
import cc.factorie.app.nlp.lemma.TokenLemma
import cc.factorie.app.nlp.load.{AnnotationType, GoldLabel, AutoLabel, DoNotLoad}
import cc.factorie.app.nlp.pos.{PennPosDomain, PennPosTag, LabeledPennPosTag}

import scala.io.Source

class GoldPos(token: Token, value: String) extends PennPosTag(token, PennPosDomain.index(value))

object LoadWSJStanfordDeps {

  private def addDepInfo(s: Sentence, depInfoSeq: Seq[(Int,Int,String)]): Unit = {
    val tree = new StanfordParseTree(s, depInfoSeq.map(_._2), depInfoSeq.map(_._3))
    s.attr += tree
  }

  def fromLines(lines:Iterator[String], filename:String = "?UNKNOWN?", loadLemma:AnnotationType = GoldLabel, loadPos:AnnotationType = GoldLabel): Seq[Document] = {
    val document: Document = new Document().setName(filename.split("/").last)
    document.annotators(classOf[Token]) = UnknownDocumentAnnotator.getClass // register that we have token boundaries
    document.annotators(classOf[Sentence]) = UnknownDocumentAnnotator.getClass // register that we have sentence boundaries
    if (loadPos != DoNotLoad) document.annotators(classOf[pos.PennPosTag]) = UnknownDocumentAnnotator.getClass // register that we have POS tags
    var sentence: Sentence = new Sentence(document)
    var depInfoSeq = new collection.mutable.ArrayBuffer[(Int,Int,String)]
    for (line <- lines) {
      if (line.trim == "") { // Sentence boundary
        if(sentence.length > 0) {
          document.appendString("\n")
          addDepInfo(sentence, depInfoSeq)
          depInfoSeq = new collection.mutable.ArrayBuffer[(Int, Int, String)]
          sentence = null
        }
      } else {
        if (sentence eq null)
          sentence = new Sentence(document) // avoids empty sentence at the end of doc
        val fields = line.split('\t')
        assert(fields.length >= 7, "Fewer than 7 fields in file "+filename+"\nOffending line:\n"+line)

        val currTokenIdx = fields(0).toInt
        val word = fields(1)

        val lemma = fields(2)

        val goldPartOfSpeech = fields(3)
        val autoPartOfSpeech = fields(4)

        val parentIdx = fields(5).toInt
        val depLabel = fields(6)

        document.appendString(" ")
        val token = new Token(sentence, word)

        // also just add the gold pos on there (for ignoring punctuation)
        token.attr += new GoldPos(token, goldPartOfSpeech)

        loadPos match{
          case GoldLabel => {token.attr += new LabeledPennPosTag(token, goldPartOfSpeech)}
          case AutoLabel => {token.attr += new LabeledPennPosTag(token, autoPartOfSpeech)}
          case DoNotLoad => {/* do nothing */}
        }



        token.attr += new TokenLemma(token, lemma)
        depInfoSeq.append((currTokenIdx, parentIdx, depLabel))
      }
    }
    if (sentence != null && sentence.length > 0) addDepInfo(sentence, depInfoSeq)

    document.sentences.foreach(s => assert(s.attr[StanfordParseTree] != null, "Parse tree null"))

    println("Loaded 1 document with "+document.sentences.size+" sentences with "+document.asSection.length+" tokens total from file "+filename)
    //    document.sentences.zipWithIndex.foreach{case(s, i) => if(s.parse == null) println(s"null parse, sentence $i")}
    //    printDocument(document)
    Seq(document)
  }

  def fromFilename(filename:String, loadLemma:AnnotationType = GoldLabel, loadPos:AnnotationType = GoldLabel): Seq[Document] = {
    fromLines(Source.fromFile(filename).getLines(), filename, loadLemma, loadPos)
  }

  def printDocument(d: Document) =
    for (s <- d.sentences)
      println(s.attr[StanfordParseTree].toString() + "\n")

  def main(args: Array[String]) =
    for (filename <- args)
      printDocument(fromFilename(filename).head)

}