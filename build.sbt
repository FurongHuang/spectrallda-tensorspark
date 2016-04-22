name := "SpectralLDA-Tensor"

version := "1.0"

scalaVersion := "2.10.6"
crossScalaVersions := Seq("2.10.6", "2.11.8")

ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }

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


libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

libraryDependencies += "com.nativelibs4java" %% "scalaxy-loops" % "0.3.4"

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.0"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"



{
  val defaultSparkVersion = "1.5.1"
  val sparkVersion =
    scala.util.Properties.envOrElse("SPARK_VERSION", defaultSparkVersion)
  val excludeHadoop = ExclusionRule(organization = "org.apache.hadoop")
  val excludeSpark = ExclusionRule(organization = "org.apache.spark")
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion excludeAll(excludeHadoop),
    "org.apache.spark" %% "spark-mllib" % sparkVersion excludeAll(excludeHadoop),
    "org.apache.spark" %% "spark-sql" % sparkVersion excludeAll(excludeHadoop)
  )
}

{
  val defaultHadoopVersion = "0.20.2-cdh3u6"
  val hadoopVersion =
    scala.util.Properties.envOrElse("SPARK_HADOOP_VERSION", defaultHadoopVersion)
  libraryDependencies += "org.apache.hadoop" % "hadoop-client" % hadoopVersion
}


resolvers ++= Seq(
  "Local Maven Repository" at Path.userHome.asFile.toURI.toURL + ".m2/repository",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/",
  "Spray" at "http://repo.spray.cc"
)

resolvers += Resolver.sonatypeRepo("public")


resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)



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