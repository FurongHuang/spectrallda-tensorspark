name := "tensorfac"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.0"//"1.4.1"
// "org.apache.spark" %% "spark-mllib" % "1.4.1"
// "org.apache.hadoop" % "hadoop-client" % "2.6.0"


libraryDependencies  ++= Seq(
  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.11.2",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  // // //"org.scalanlp" %% "breeze-natives" % "0.11.2",
  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

libraryDependencies += "org.apache.spark" % "spark-streaming_2.10" % "1.1.0"//"1.4.1"
libraryDependencies += "org.apache.commons" % "commons-math3" % "3.0"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"
resolvers += Resolver.sonatypeRepo("public")

libraryDependencies += "log4j" % "log4j" % "1.2.14"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.0"
resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)