name := "SpectralLDA-Tensor"

version := "1.0"

scalaVersion := "2.10.6"
crossScalaVersions := Seq("2.10.6", "2.11.8")


scalacOptions :=  Seq(
  "-unchecked",
  "-feature",
  "-Xlint",
  "-Ywarn-dead-code",
  "-Ywarn-adapted-args",
  "-Ywarn-numeric-widen",
  "-Ywarn-value-discard",
  "-target:jvm-1.7",
  "-encoding", "UTF-8",
  "-optimise",
  "-Yclosure-elim",
  "-Yinline"
)

// Spark relies on a certain version of breeze, we avoid interfering
// with the built-in version, which could otherwise break down
// the code

// If from a certain version, Spark no longer relies on breeze, we need
// to activate the following lines

//libraryDependencies ++= Seq(
//  "org.scalanlp" %% "breeze" % "[0.11.2,)",
//  "org.scalanlp" %% "breeze-natives" % "[0.11.2,)"
//)

libraryDependencies ++= Seq(
    "com.nativelibs4java" %% "scalaxy-loops" % "[0.3.4,)",
    "com.github.scopt" %% "scopt" % "[3.4.0,)",
    "org.scalatest" %% "scalatest" % "[2.2.6,)" % "test",
    "org.scalatest" %% "scalatest-matchers" % "[2.2.6,)" % "test"
)


{
  val defaultSparkVersion = "[1.6.1,)"
  val sparkVersion =
    scala.util.Properties.envOrElse("SPARK_VERSION", defaultSparkVersion)

  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
  )
}

//{
//  val defaultHadoopVersion = "[2.6.0,)"
//  val hadoopVersion =
//    scala.util.Properties.envOrElse("SPARK_HADOOP_VERSION", defaultHadoopVersion)
//  libraryDependencies += "org.apache.hadoop" % "hadoop-client" % hadoopVersion
//}

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
  {
    case PathList("javax", "servlet", xs @ _*)               => MergeStrategy.first
    case PathList(ps @ _*) if ps.last endsWith ".html"       => MergeStrategy.first
    case "application.conf"                                  => MergeStrategy.concat
    case "reference.conf"                                    => MergeStrategy.concat
    case "log4j.properties"                                  => MergeStrategy.first
    case m if m.toLowerCase.endsWith("manifest.mf")          => MergeStrategy.discard
    case m if m.toLowerCase.matches("meta-inf.*\\.sf$")      => MergeStrategy.discard
    case m if m.toLowerCase.startsWith("meta-inf/services/") => MergeStrategy.filterDistinctLines
    case _ => MergeStrategy.first
  }
}
