package edu.umass.cs.iesl.nndepparse

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
  val numToPrecompute = new CmdOption("precompute-words", -1, "INT", "Number of word embeddings to precompute")
}