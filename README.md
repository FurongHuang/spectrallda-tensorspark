# Spectral LDA on Spark

## Summary 
* This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark.
* We also implemented a Spectral LDA model via sketching to accelerate tensor building and decomposition.
* Version: 1.0

## How do I get set up?
We use the `sbt` build system. By default we support Scala 2.11.8 and Spark 2.0.0 upward. Cross build to Scala 2.10.6 is also supported. The documentation below supposes we're using Scala 2.11.

### To run from the command line
1. First compile and package the entire repo.

    ```bash
    sbt package
    ```
    
    It will produce `target/scala-2.11/spectrallda-tensor_2.11-1.0.jar`.
    
2. The command line usage is 
    
    ```bash
    Spectral LDA Factorization
    Usage: SpectralLDA [options] <input>...
    
      -k, --k <value>          number of topics
      -alpha0, --topicConcentration <value>
                               the sum of the prior vector for topic distribution e.g. k for a non-informative prior.
      -idf, --idfLowerBound <value>
                               only work on terms with IDF above the lower bound. default: 1.0
      --M2-cond <value>        stop if the M2 condition number is higher than the given bound. default: 50.0
      -max-iter, --maxIterations <value>
                               number of iterations of learning. default: 200
      --sketching              Tensor decomposition via sketching
      -B, --B <value>          number of hash families for sketching. default: 50
      -b, --b <value>          length of a hash for sketching, preferably to be power of 2. default: 256
      -t, --inputType <value>  type of input files: "obj", "libsvm" or "text". "obj" for Hadoop SequenceFile of RDD[(Long, SparseVector[Double])]. default: obj
      -o, --outputDir <dir>    output write path. default: .
      --stopWordFile <value>   filepath for a list of stopwords. default: src/main/resources/Data/datasets/StopWords_common.txt
      --help                   prints this usage text
      <input>...               paths of input files    
    ```
    
    The parameters `-k`, `-alpha0` and the input file paths are required, the others are optional.
    
    A good choice for `alpha0` is equal to `k` so that we have a non-informative Dirichilet prior for the topic distribution -- any topic distribution is equally likely.
    
    `--M2-cond` checks the shifted M2 condition number (the ratio of the maximum eigenvalue to the minimum one) and stops if it's above the given bound.
    
    We could specify `--sketching` to do the sketching-based decomposition, which is off by default. The associated parameters are the number of hash families `B` (default to 50) and length of a single hash `b` (default to 2^8). These values are a good choice to start with.
    
    The file type `-t` could be "text", "libsvm", or "obj": "text" for plain text files, "libsvm" for text files in LIBSVM format, "obj" for Hadoop SequenceFiles storing serialised `RDD[(Long, SparseVector[Double])]`. It is "obj" by default.
    
3. An example call from command line is

    ```bash
    spark-submit --packages com.github.scopt:scopt_2.11:3.5.0 \
    --class edu.uci.eecs.spectralLDA.SpectralLDA \
    target/scala-2.11/spectrallda-tensor_2.11-1.0.jar \
    -k 5 -alpha0 5.0 -t libsvm -o results --sketching \
    src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt
    ```
    
    It runs with `alpha0=k=5`, enables the sketching, specifies the input file in LIBSVM format, and outputs results in `result/`.
    
### API usage
For sketching-based decomposition, below is an example snippet.

```scala
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import edu.uci.eecs.spectralLDA.algorithm.TensorLDASketch
import breeze.linalg._

// The sketcher that hashes a tensor into B-by-b matrix,
// where B is the number of hash families, b is the length of
// a single hash
val sketcher = TensorSketcher[Double, Double](
  n = Seq(params.k, params.k, params.k),
  B = params.B,
  b = params.b
)

// The sketching-based fitting algorithm 
val lda = new TensorLDASketch(
  dimK = params.k,
  alpha0 = params.topicConcentration,
  sketcher = sketcher,
  idfLowerBound = value,                     // optional, default: 1.0
  m2ConditionNumberUB = value,               // optional, default: infinity
  maxIterations = 200,                       // optional, default: 200
  randomisedSVD = true                       // optional, default: true
)(tolerance = params.tolerance)              // optional, default: 1e-9

// Fit against the documents
// beta is the V-by-k matrix, where V is the vocabulary size, 
// k is the number of topics. It stores the word distribution 
// per topic column-wise
// alpha is the length-k Dirichlet prior for the topic distribution
val (beta: DenseMatrix[Double], alpha: DenseVector[Double]) = lda.fit(documents)
```

For non-sketching-based decomposition, the usage is simpler.

```scala
import edu.uci.eecs.spectralLDA.algorithm.TensorLDA
import breeze.linalg._

val lda = new TensorLDA(
  dimK = params.k,
  alpha0 = params.topicConcentration,
  maxIterations = params.maxIterations,  // optional, default 200
  tolerance = params.tolerance           // optional, default 1e-9
)

// If we want to only work on terms with IDF above a certain bound
// import edu.uci.eecs.spectralLDA.textprocessing.TextProcessor
// val filteredDocs = TextProcessor.filterIDF(documents, <IDF lower bound>)

// Fit against the documents
// beta is the V-by-k matrix, where V is the vocabulary size, 
// k is the number of topics. It stores the word distribution 
// per topic column-wise
// alpha is the length-k Dirichlet prior for the topic distribution
val (beta: DenseMatrix[Double], alpha: DenseVector[Double]) = lda.fit(documents)
```

### Set up Spark 2.0.0 to use system native BLAS/LAPACK

1. In order for Spark to use system native BLAS/LAPACK, first compile Spark 2.0.0 with the option `-Pnetlib-lgpl` to include all the artifacts of `netlib4java`, following the advice [here](http://apache-spark-user-list.1001560.n3.nabble.com/Mllib-native-netlib-java-OpenBLAS-td19662.html).

    ```bash
    mvn -Pyarn -Phadoop-2.7 -Pnetlib-lgpl -DskipTests clean package
    ```

    `netlib4java` includes the JNI routines to load up the system native BLAS/LAPACK libraries. 

2. Now we're going to make the system native BLAS/LAPACK libraries available to `netlib4java`. On Mac, `netlib4java` will automatically find `veclib`; on Linux, we could use ATLAS.

3. Lastly set up symbollic links for the `libblas.so.3` and `liblapack.so.3` that `netlib4java` looks for. 

    ```bash
    sudo alternatives --install /usr/lib64/libblas.so libblas.so /usr/lib64/atlas/libtatlas.so.3 1000
    sudo alternatives --install /usr/lib64/libblas.so.3 libblas.so.3 /usr/lib64/atlas/libtatlas.so.3 1000
    sudo alternatives --install /usr/lib64/liblapack.so liblapack.so /usr/lib64/atlas/libtatlas.so.3 1000
    sudo alternatives --install /usr/lib64/liblapack.so.3 liblapack.so.3 /usr/lib64/atlas/libtatlas.so.3 1000
    ```

Now if we run the above experiments again, any "WARN BLAS" or "WARN LAPACK" messages should have disappeared.

### I have millions of small text files...
If we open them simply via `sc.wholeTextFiles()` the system will spend forever long time querying the file system for the list of all the file names. The solution is to first combine them in Hadoop SequenceFiles of `RDD[(String, String)]`, then process them into word count vectors and vocabulary array.

1. We provided `edu.uci.eecs.spectralLDA.textprocessing.CombineSmallTextFiles` to squash many text files into a Hadoop SequenceFile. For example, all the Wikipedia articles are extracted under `wikitext/0` to `wikitext/9999`, with each subdirectory containing thousands of text files.

    ```bash
    # Under wikitext/, first list all the subdirectory names,
    # then call xargs to feed, say 50 subdirectories each time to CombineSmallTextFiles
    find . -mindepth 1 -maxdepth 1 | xargs -n 50 \
    spark-submit --class edu.uci.eecs.spectralLDA.textprocessing.CombineSmallTextFiles \
    target/scala-2.11/spectrallda-tensor_2.11-1.0.jar
    ```
    
    When the loop finishes, we'd find many `*.obj` Hadoop SequenceFiles under `wikitext/`.
    
2. Within `sbt console`, we process the SequenceFiles into word count vectors `RDD[(Long, SparseVector[Double])]` and dictionary array, and save them. 

    ```scala
    import org.apache.spark.{SparkConf, SparkContext}
    import org.apache.spark.rdd.RDD
    import edu.uci.eecs.spectralLDA.textprocessing.TextProcessor
    val conf = new SparkConf().setAppName("Word Count")
    val sc = new SparkContext(conf)
    val (docs, dictionary) = TextProcessor.processDocumentsRDD(
      sc.objectFile("wikitext/*.obj"),
      stopwordFile = "src/main/resources/Data/datasets/StopWords_common.txt",
      vocabSize = <vocabSize>
    )
    docs.saveAsObjectFile("docs.obj")
    ```
    
    The output file `docs.obj` contains serialised `RDD[(Long, SparseVector[Double])]`. When we run `SpectralLDA` later on, we could specify the input file `docs.obj` and the file type as `obj`.

    
## References
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* Fast and Guaranteed Tensor Decomposition via Sketching: http://arxiv.org/abs/1506.04448
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html

## Who do I talk to?

* Repo owner or admin: Furong Huang 
* Contact: furongh.uci@gmail.com
