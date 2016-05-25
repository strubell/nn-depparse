
organization := "edu.umass.cs.iesl"

name := "nn-depparse"

version := "0.1-SNAPSHOT"

scalaVersion := "2.11.7"

resolvers += "IESL snapshot repository" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/snapshots/"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "cc.factorie" %% "factorie" % "1.2-SNAPSHOT",
  "cc.factorie.app.nlp" % "all-models" % "1.2-SNAPSHOT",
  "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test",
  "junit" % "junit" % "4.11" % "test",
  "com.novocode" % "junit-interface" % "0.9" % "test->default",
  "sis" % "jhdf5_2.11" % "14.2.5-r35810"
//  "edu.umass.cs.iesl" % "lffi" % "0.1-SNAPSHOT"
)


lazy val execScript = taskKey[Unit]("Execute script to install jhdf5")

execScript := {
  s"${baseDirectory.value}/bin/install-jhdf5.sh" !
}

parallelExecution := true

compile in Compile := {
  execScript.value
  (compile in Compile).value
}
