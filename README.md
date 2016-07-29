# Spectral LDA on Spark

## Summary 
* This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark.
* We also implemented a Spectral LDA model via sketching to accelerate tensor building and decomposition.
* Version: 1.0

## How do I get set up?
We use the `sbt` build system. By default we support Scala 2.11.8 and Spark 2.0.0 upward. In `build.sbt` we also support cross build to Scala 2.10.6.

For example, the first line below will compile and package the code using Scala 2.11.8 as it's the default setup. The second line uses Scala 2.10.6 as we explicitly called for it. If we precede any `sbt` command by the `+` sign as does the third line, both the Scala 2.10 and 2.11 based versions will be produced.

```bash
sbt package
sbt ++2.10.6 package
sbt "+ package"
```

### To build the project
1. install `sbt`
2. open your terminal:

    ```bash
    cd SpectralLDA-TensorSpark
    sbt "+ assembly"
    ```    
    
    This will pack the class files and all the dependencies into a single fat JAR file for both Scala 2.10 and 2.11. The path to the jar is: `<PROJECT-PATH>/target/scala-<ver>/SpectralLDA-Tensor-assembly-1.0.jar`
3. deploy the application using [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html).  

### Let's run some experiments

1. Synthetic Experiments:
    ```bash
    ./bin/spark-submit --class edu.uci.eecs.spectralLDA.SpectralLDA \
    --master local[2] --deploy-mode client \
    <PROJECT-PATH>/target/scala-<ver>/SpectralLDA-Tensor-assembly-1.0.jar \
    <PROJECT-PATH>/src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt \
    --libsvm 1 
    ```
    1. Data generation script in MATLAB is provided in the repository [here](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/data/SyntheticDataGenerator.m?at=master&fileviewer=file-view-default). One can play around with hyperparameters such as Sample Size, Vocabulary Size, Hidden Dimension, and How mixed the topics are.  The synthetic data for training are then generated as `datasets/synthetic/samples_train_libsvm.txt` and `datasets/synthetic/samples_test_libsvm.txt` in the libsvm format and as `datasets/synthetic/samples_train_DOK.txt` and `datasets/synthetic/samples_test_DOK.txt` in the DOK format. 
    2. Our program reads libsvm format.

2. Real Experiments:
    ```bash
    ./bin/spark-submit --class edu.uci.eecs.spectralLDA.SpectralLDA \
    --master local[2] --deploy-mode client \
    <PROJECT-PATH>/target/scala-<ver>/SpectralLDA-Tensor-assembly-1.0.jar \
    <PROJECT-PATH>/src/main/resources/Data/datasets/enron_email/corpus.txt
    ```
    1. Our program takes raw text (NOTE: Each text file line should hold 1 document). 

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
    # then call xargs to feed, say 50 subdirectories to 
    find . -mindepth 1 -maxdepth 1 | xargs -n 50 \
    spark-submit --class edu.uci.eecs.spectralLDA.textprocessing.CombineSmallTextFiles \
    target/scala-2.11/spectrallda-tensor_2.11-1.0.jar
    ```
    
    When the loop finishes, we'd find many `*.obj` Hadoop SequenceFiles under `wikitext/`.
    
2. Within `sbt console`, we process the SequenceFiles into word count vectors `RDD[(Long, SparseVector[Double])]` and dictionary array, and save them. 

    ```bash
    sbt console
    scala> import org.apache.spark.{SparkConf, SparkContext}
    scala> import org.apache.spark.rdd.RDD
    scala> import edu.uci.eecs.spectralLDA.textprocessing.TextProcessor
    scala> val conf = new SparkConf().setAppName("Word Count")
    scala> val sc = new SparkContext(conf)
    scala> val (docs, dictionary) = TextProcessor.processDocumentsRDD(
    scala>   sc.objectFile("wikitext/*.obj"),
    scala>   stopwordFile = "src/main/resources/Data/datasets/StopWords_common.txt",
    scala>   vocabSize = <vocabSize>
    scala> )
    scala> docs.saveAsObjectFile("docs.obj")
    ```
    
    The output file `docs.obj` contains serialised `RDD[(Long, SparseVector[Double])]`. When we run `SpectralLDA` later on, we could specify the input file `docs.obj` and the file type as `obj`.

    
## References
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* Fast and Guaranteed Tensor Decomposition via Sketching: http://arxiv.org/abs/1506.04448
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html

## Who do I talk to?

* Repo owner or admin: Furong Huang 
* Contact: furongh.uci@gmail.com
