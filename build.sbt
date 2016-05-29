
organization := "edu.umass.cs.iesl"

name := "nn-depparse"

version := "0.1-SNAPSHOT"

scalaVersion := "2.11.7"

resolvers += "IESL snapshot repository" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/snapshots/"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "cc.factorie" %% "factorie" % "1.2-SNAPSHOT",
  "cc.factorie.app.nlp" % "all-models" % "1.2-SNAPSHOT",
  "sis" % "jhdf5_2.11" % "14.2.5-r35810"
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
