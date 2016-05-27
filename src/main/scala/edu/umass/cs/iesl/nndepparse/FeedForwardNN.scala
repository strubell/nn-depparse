package edu.umass.cs.iesl.nndepparse

import cc.factorie.la.{DenseTensor1, DenseTensor2}
import ch.systemsx.cisd.hdf5.HDF5Factory

class FeedForwardNN(modelFile: String, featureFunc: (ParseState, Array[Int], Array[Int], Array[Int]) => Unit, topKPrecompute: Int = -1) {

  println(s"Loading model from $modelFile")
  val reader = HDF5Factory.openForReading(modelFile)

  val wordEmbeddings = reader.readDoubleMatrix("word_embeddings")
  val posEmbeddings = reader.readDoubleMatrix("pos_embeddings")
  val labelEmbeddings = reader.readDoubleMatrix("label_embeddings")

  val wordHidden = new DenseTensor2(reader.readDoubleMatrix("word_hidden"))
  val posHidden = new DenseTensor2(reader.readDoubleMatrix("pos_hidden"))
  val labelHidden = new DenseTensor2(reader.readDoubleMatrix("label_hidden"))

  val wordBias = new DenseTensor1(reader.readDoubleArray("word_bias"))
  val posBias = new DenseTensor1(reader.readDoubleArray("pos_bias"))
  val labelBias = new DenseTensor1(reader.readDoubleArray("label_bias"))


  val outputLayer = new DenseTensor2(reader.readDoubleMatrix("output_layer"))
  val outputBias = new DenseTensor1(reader.readDoubleArray("output_bias"))

  reader.close()

  println(s"wordEmbeddings: (${wordEmbeddings.length},${wordEmbeddings(0).length})")
  println(s"posEmbeddings: (${posEmbeddings.length},${posEmbeddings(0).length})")
  println(s"labelEmbeddings: (${labelEmbeddings.length},${labelEmbeddings(0).length})")

  println(s"wordHidden: (${wordHidden.dim1},${wordHidden.dim2})")
  println(s"posHidden: (${posHidden.dim1},${posHidden.dim2})")
  println(s"labelHidden: (${labelHidden.dim1},${labelHidden.dim2})")

  println(s"output: (${outputLayer.dim1},${outputLayer.dim2})")

  val posDomainSize = posEmbeddings.length
  val wordDomainSize = wordEmbeddings.length
  val labelDomainSize = labelEmbeddings.length


  val wordEmbeddingSize = wordEmbeddings(0).length
  val posEmbeddingSize = posEmbeddings(0).length
  val labelEmbeddingSize = labelEmbeddings(0).length

  val numWordEmbeddings = wordHidden.dim1/wordEmbeddingSize
  val numPosEmbeddings = posHidden.dim1/posEmbeddingSize
  val numLabelEmbeddings = labelHidden.dim1/labelEmbeddingSize

  val hiddenLayerSize = wordHidden.dim2

  // precompute pos hiddens
  val posPrecomputed = Array.fill(numPosEmbeddings)(new Array[DenseTensor1](posDomainSize))
  var i = 0
  while(i < numPosEmbeddings){
    var j = 0
    val start = i*posEmbeddingSize
    val end = start+posEmbeddingSize
    while(j < posDomainSize) {
      val embedded = leftMultiplyToFrom(posHidden, start, end, new DenseTensor1(posEmbeddings(j)))
        posPrecomputed(i)(j) = embedded
      j += 1
    }
    i += 1
  }

  // precompute label hiddens
  val labelPrecomputed = Array.fill(numLabelEmbeddings)(new Array[DenseTensor1](labelDomainSize))
  i = 0
  while(i < numLabelEmbeddings){
    var j = 0
    val start = i*labelEmbeddingSize
    val end = start+labelEmbeddingSize
    while(j < labelDomainSize) {
      val embedded = leftMultiplyToFrom(labelHidden, start, end, new DenseTensor1(labelEmbeddings(j)))
      labelPrecomputed(i)(j) = embedded
      j += 1
    }
    i += 1
  }

  // precompute first topKPrecompute word hiddens (or all if topKPrecompute == -1)
  val topK = if(topKPrecompute < 0) wordDomainSize else topKPrecompute
  val wordPrecomputed = Array.fill(numWordEmbeddings)(new Array[DenseTensor1](topK))
  i = 0
  while(i < numWordEmbeddings){
    var j = 0
    val start = i*wordEmbeddingSize
    val end = start+wordEmbeddingSize
    while(j < topK) {
      val embedded = leftMultiplyToFrom(wordHidden, start, end, new DenseTensor1(wordEmbeddings(j)))
      wordPrecomputed(i)(j) = embedded
      j += 1
    }
    i += 1
  }

  def add(tensors: DenseTensor1*) = {
    val acc = tensors(0)
    var i = 1
    while(i < tensors.length){
      acc += tensors(i)
      i += 1
    }
    acc
  }

  def leftMultiplyToFrom(mat: DenseTensor2, start: Int, end: Int, vec: DenseTensor1): DenseTensor1 = {
    val res = new DenseTensor1(mat.dim2)
    var i = start
    while (i < end) {
      var j = 0
      while (j < mat.dim2) {
        res(j) += mat(i, j) * vec(i-start)
        j += 1
      }
      i += 1
    }
    res
  }

  def leftMultiplyReluWithBias(mat: DenseTensor2, vec: DenseTensor1, bias: DenseTensor1): DenseTensor1 = {
    val res = new DenseTensor1(bias)
    var i = 0
    while (i < mat.dim1) {
      if(vec(i) > 0) {
        var j = 0
        while (j < mat.dim2) {
          res(j) += mat(i, j) * vec(i)
          j += 1
        }
      }
      i += 1
    }
    res
  }

  def accumulatePrecomputedPartialWithBias(intFeats: Array[Int], precomputed: Array[Array[DenseTensor1]], bias: DenseTensor1): DenseTensor1 = {
    var i,j = 0
    val acc = new DenseTensor1(bias)
    val accArr = acc.asArray
    while(i < intFeats.length){
      val featIdx = intFeats(i) - 1
      if(featIdx < topK) {
        j = 0
        while (j < acc.length) {
          accArr(j) += precomputed(i)(featIdx)(j)
          j += 1
        }
      }
      else {
        val start = i*wordEmbeddingSize
        val end = start+wordEmbeddingSize
        acc += leftMultiplyToFrom(wordHidden, start, end, new DenseTensor1(wordEmbeddings(featIdx)))
      }
      i += 1
    }
    acc
  }

  def accumulatePrecomputedWithBias(intFeats: Array[Int], precomputed: Array[Array[DenseTensor1]], bias: DenseTensor1): DenseTensor1 = {
    var i = 0
    val acc = new DenseTensor1(bias)
    while(i < intFeats.length){
      acc += precomputed(i)(intFeats(i)-1)
      i += 1
    }
    acc
  }

  def predict(state: ParseState): String = {

    val wordInput = new DenseTensor1(wordHidden.dim1)
    val posInput = new DenseTensor1(posHidden.dim1)
    val labelInput = new DenseTensor1(labelHidden.dim1)

    val wordIntFeats = new Array[Int](numWordEmbeddings)
    val posIntFeats = new Array[Int](numPosEmbeddings)
    val labelIntFeats = new Array[Int](numLabelEmbeddings)

    // compute features
    featureFunc(state, wordIntFeats, posIntFeats, labelIntFeats)

//    println(s"feats: ${wordIntFeats.sum} ${posIntFeats.sum} ${labelIntFeats.sum}")

    val wordOutput = accumulatePrecomputedWithBias(wordIntFeats, wordPrecomputed, wordBias)
//      if(topKPrecompute == -1)
//        accumulatePrecomputedWithBias(wordIntFeats, wordPrecomputed, wordBias)
//      else
//        accumulatePrecomputedPartialWithBias(wordIntFeats, wordPrecomputed, wordBias)

    val posOutput = accumulatePrecomputedWithBias(posIntFeats, posPrecomputed, posBias)

    val labelOutput = accumulatePrecomputedWithBias(labelIntFeats, labelPrecomputed, labelBias)

//    println(s"word: ${wordOutput.oneNorm} pos: ${posOutput.oneNorm} label: ${labelOutput.oneNorm}")

    val combined = add(wordOutput, posOutput, labelOutput)

//    println(s"add: ${combined.oneNorm}")

    val scores = leftMultiplyReluWithBias(outputLayer, combined, outputBias)

    val argmax = scores.maxIndex
    IntMaps.intToDecisionMap(argmax+1)
  }

}