# Spectral LDA on Spark

## Summary 
* This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark.
* We also implemented a Spectral LDA model via sketching to accelerate tensor building and decomposition.
* Version: 1.0

## How do I get set up?
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

### Set up Spark to use system native BLAS/LAPACK

1. In order for Spark to use system native BLAS/LAPACK, first compile Spark with the option `-Pnetlib-lgpl` to include all the artifacts of `netlib4java`, following the advice [here](http://apache-spark-user-list.1001560.n3.nabble.com/Mllib-native-netlib-java-OpenBLAS-td19662.html). E.g for Scala 2.11, we could do

    ```bash
    dev/change-scala-version.sh 2.11
    mvn -Pyarn -Phadoop-2.6 -Pnetlib-lgpl -Dscala-2.11 -DskipTests clean package
    ```

    `netlib4java` includes the JNI routines to load up the system native BLAS/LAPACK libraries. 

2. Now we're going to make the system native BLAS/LAPACK libraries available to `netlib4java`. On Mac, `netlib4java` will automatically find `veclib`; on Linux, the best option is to compile a copy of OpenBLAS as it provides the complete BLAS/CBLAS/LAPACK that `netlib4java` needs and it boasts of the best performance vs BLAS/ATLAS. Following the instructions on [https://github.com/xianyi/OpenBLAS/](https://github.com/xianyi/OpenBLAS/), it's as simple as 

    ```bash
    # Make sure gfortran is installed,
    # otherwise LAPACK routines won't get compiled

    make
    sudo make install
    ```

    By default it's installed into `/opt/OpenBLAS`. Inside `/opt/OpenBLAS/lib` lies the newly-compiled library `libopenblas.so`!

3. The last thing to do is to set up some symbollic links. `netlib4java` searches for and loads `libblas.so.3` and `liblapack.so.3`. We'll create system `alternatives` links associating these two modules to the `libopenblas.so`. If the system-default `.so` modules reside in `/usr/lib64`, do

    ```bash
    sudo update-alternatives --install /usr/lib64/libblas.so libblas.so /opt/OpenBLAS/lib/libopenblas.so 1000
    sudo update-alternatives --install /usr/lib64/libblas.so.3 libblas.so.3 /opt/OpenBLAS/lib/libopenblas.so 1000
    sudo update-alternatives --install /usr/lib64/liblapack.so liblapack.so /opt/OpenBLAS/lib/libopenblas.so 1000
    sudo update-alternatives --install /usr/lib64/liblapack.so.3 liblapack.so.3 /opt/OpenBLAS/lib/libopenblas.so 1000
    ```

Now if we run the above experiments again, any "WARN BLAS" or "WARN LAPACK" messages should have disappeared.

**NOTE:** In Breeze the feature of sparse matrix is still experimental and it may not work with native BLAS/LAPACK, in this case use `alternatives --remove` to stop exposing the native libs to Spark. In our code `DataCumulant` and `DataCumulantSketch` will use heavily sparse matrices when computing M2 if we specify `randomisedSVD = true`.

### Dependencies

* You should install `sbt` before running this program.
* See `build.sbt` for the dependencies.


## References
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* Fast and Guaranteed Tensor Decomposition via Sketching: http://arxiv.org/abs/1506.04448
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html

## Who do I talk to?

* Repo owner or admin: Furong Huang 
* Contact: furongh@uci.edu
