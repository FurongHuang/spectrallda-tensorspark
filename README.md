# Spectral LDA on Spark

## Summary 
* This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark.
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
      -t, --inputType <value>  type of input files: "obj", "libsvm" or "text". "obj" for Hadoop SequenceFile of RDD[(Long, SparseVector[Double])]. default: obj
      -o, --outputDir <dir>    output write path. default: .
      --stopWordFile <value>   filepath for a list of stopwords. default: src/main/resources/Data/datasets/StopWords_common.txt
      --help                   prints this usage text
      <input>...               paths of input files    
    ```
    
    The parameters `-k`, `-alpha0` and the input file paths are required, the others are optional.
    
    A good choice for `alpha0` is equal to `k` so that we have a non-informative Dirichilet prior for the topic distribution -- any topic distribution is equally likely.
    
    `--M2-cond` checks the condition number (the ratio of the maximum eigenvalue to the minimum one) of the shifted M2 matrix and stops if it's above the given bound. It allows to quickly check if there's any predominant topic in the input.
    
    The file type `-t` could be "text", "libsvm", or "obj": "text" for plain text files, "libsvm" for text files in LIBSVM format, "obj" for Hadoop SequenceFiles storing serialised `RDD[(Long, SparseVector[Double])]`. It is "obj" by default.
    
3. An example call from command line is

    ```bash
    spark-submit --packages com.github.scopt:scopt_2.11:3.5.0 \
    --class edu.uci.eecs.spectralLDA.SpectralLDA \
    target/scala-2.11/spectrallda-tensor_2.11-1.0.jar \
    -k 5 -alpha0 5.0 -t libsvm -o results \
    src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt
    ```
    
    It runs with `alpha0 = k = 5`, specifies the input file in LIBSVM format, and outputs results in `result/`.
    
### API usage
The API is designed following the lines of the Spark built-in `LDA` class.

```scala
import edu.uci.eecs.spectralLDA.algorithm.TensorLDA
import breeze.linalg._

val lda = new TensorLDA(
  dimK = params.k,
  alpha0 = params.topicConcentration,
  maxIterations = value,            // optional, default: 200
  idfLowerBound = value,            // optional, default: 1.0
  m2ConditionNumberUB = value,      // optional, default: infinity
  randomisedSVD = true              // optional, default: true
)(tolerance = params.tolerance)     // optional, default: 1e-9

// Fit against the documents
// beta is the V-by-k matrix, where V is the vocabulary size, 
// k is the number of topics. It stores the word distribution 
// per topic column-wise
// alpha is the length-k Dirichlet prior for the topic distribution
// eigvecM2 is the V-by-k matrix for the top k eigenvectors of M2
// eigvalM2 is the length-k vector for the top k eigenvalues of M2
// m1 is the length-V vector for the average word distribution
val (beta: DenseMatrix[Double], alpha: DenseVector[Double], 
  eigvecM2: DenseMatrix[Double], eigvalM2: DenseVector[Double],
  m1: DenseVector[Double]) = lda.fit(documents)
```

If one just wants to decompose a 3rd-order symmetric tensor into the sum of rank-1 tensors, we could do

```scala
import edu.uci.eecs.spectralLDA.algorithm.ALS
import breeze.linalg._

val als = new ALS(
  dimK = value,
  thirdOrderMoments = value,        // k-by-(k*k) matrix for the unfolded 3rd-order symmetric tensor
  maxIterations = value             // optional, default: 200
)

// We run ALS to find the best approximating sum of rank-1 tensors such that 
// $$ M3 = \sum_{i=1}^k\alpha_i\beta_i^{\otimes 3} $$

// beta is the k-by-k matrix with $\beta_i$ as columns
// alpha is the vector for $(\alpha_1,\ldots,\alpha_k)$
val (beta: DenseMatrix[Double], _, _, alpha: DenseVector[Double]) = als.run
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

1. We provided `edu.uci.eecs.spectralLDA.textprocessing.CombineSmallTextFiles` to squash many text files into a Hadoop SequenceFile. As an example, if all the Wikipedia articles are extracted under `wikitext/0` to `wikitext/9999`, with thousands of text files under each subdirectory, we could batch combining them into a series of SequenceFiles.

    ```bash
    # Under wikitext/, first list all the subdirectory names,
    # then call xargs to feed, say 50 subdirectories each time to CombineSmallTextFiles
    find . -mindepth 1 -maxdepth 1 | xargs -n 50 \
    spark-submit --class edu.uci.eecs.spectralLDA.textprocessing.CombineSmallTextFiles \
    target/scala-2.11/spectrallda-tensor_2.11-1.0.jar
    ```
    
    When the loop finishes, we'd find a number of `*.obj` Hadoop SequenceFiles under `wikitext/`.
    
2. Launch `spark-shell` with the proper server and memory settings, and the option `--jars target/scala-2.11/spectrallda-tensor_2.11-1.0.jar`. 

   We process the SequenceFiles into word count vectors `RDD[(Long, SparseVector[Double])]` and dictionary array, and save them. 

    ```scala
    import org.apache.spark.rdd.RDD
    import edu.uci.eecs.spectralLDA.textprocessing.TextProcessor
    val (docs, dictionary) = TextProcessor.processDocumentsRDD(
      sc.objectFile[(String, String)]("wikitext/*.obj"),
      stopwordFile = "src/main/resources/Data/datasets/StopWords_common.txt",
      vocabSize = <vocabSize>
    )
    docs.saveAsObjectFile("docs.obj")
    ```
    
    The output file `docs.obj` contains serialised `RDD[(Long, SparseVector[Double])]`. When we run `SpectralLDA` later on, we could specify the input file `docs.obj` and the file type as `obj`.

    
## References
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html

## Who do I talk to?

* Repo owner or admin: Furong Huang 
* Contact: furongh.uci@gmail.com
