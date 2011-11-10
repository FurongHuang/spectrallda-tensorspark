name := "SpectralLDA-Tensor"

version := "1.0"

scalaVersion := "2.10.5"
crossScalaVersions := Seq("2.10.5", "2.11.7")

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

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.1"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

libraryDependencies += "com.nativelibs4java" %% "scalaxy-loops" % "0.3.4"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "1.5.1"

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.1"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"
