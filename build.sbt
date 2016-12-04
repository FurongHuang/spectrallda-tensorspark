name := "SpectralLDA-Tensor"

version := "1.0"

scalaVersion := "2.11.8"
crossScalaVersions := Seq("2.10.6", "2.11.8")

publishMavenStyle := true


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


spName := "FurongHuang/spectrallda-tensorspark"

sparkVersion := "2.0.2"

sparkComponents += "mllib"

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

spShortDescription := "Spectral LDA Algorithm using Tensor Decomposition"

spDescription := """The Spectral LDA algorithm transforms the LDA problem to an orthogonal tensor decomposition problem. [[Anandkumar 2012]] establishes theoretical guarantee for the convergence of orthogonal tensor decomposition. 

This algorithm first builds 2nd-order, 3rd-order moments from the empirical word counts, orthogonalize them and finally perform the tensor decomposition on the orthogonalised data moments. The whole procedure is purely linear and could leverage machine native BLAS/LAPACK libraries (the Spark needs to be compiled with `-Pnetlib-lgpl` option).

It achieves competitive log-perplexity vs Online Variational Inference in the shortest time. It also has clean memory usage -- as of v2.0.0 we've experienced crash due to memory problem with the built-in Gibbs Sampler or Online Variational Inference, but never with the Spectral LDA algorithm. This algorithm is linearly scalable. 

The official repo is at https://github.com/FurongHuang/SpectralLDA-TensorSpark. We refactored for the Spark coding style and interfaces when porting over for the PR. We wrote a report describing the algorithm in detail and listing test results at https://www.overleaf.com/read/wscdvwrjmtmw.

REFERENCES
Tensor decompositions for learning latent variable models, by A. Anandkumar, R. Ge, D. Hsu, S.M. Kakade and M. Telgarsky. JMLR 2014.

Online Tensor Methods for Learning Latent Variable Models, by F. Huang, U.N.Niranjan, M.U.Hakeem, A.Anandkumar. JMLR 2014."""


licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")



