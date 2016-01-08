# README #
* Quick summary: 
This code implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark. 

* Version: 0.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### Run a Synthetic Experiment Toy Example
* install sbt
* open your terminal
* cd to SpectralLDA-Tensor
* type: "sbt" 
* type: "assembly". This will pack the class files and all the dependencies into a single fat JAR file. The path to the jar is: <PROJECT-PATH>/SpectralLDA-TensorSpark/SpectralLDA-Tensor/target/scala-2.10/SpectralLDA-Tensor-assembly-1.0.jar
* deploy the application using [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html).  



### How do I get set up? ###


* Summary of set up
Main file is SpectralLDA-Tensor/src/main/scala-2.10/SpectralLDA.scala

* Configuration 

* (1). Synthetic Experiments:

> ./bin/spark-submit --class edu.uci.eecs.spectralLDA.SpectralLDA --master local[2] --deploy-mode client <PROJECT-PATH>/SpectralLDA-TensorSpark/SpectralLDA-Tensor/target/scala-2.10/SpectralLDA-Tensor-assembly-1.0.jar <PROJECT-PATH>/SpectralLDA-TensorSpark/src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt --synthetic 1 


*    (1.1).  Data generation script in MATLAB is provided in the repository [here](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/data/SyntheticDataGenerator.m?at=master&fileviewer=file-view-default). One can play around with hyperparameters such as Sample Size, Vocabulary Size, Hidden Dimension, and How mixed the topics are.  The synthetic data for training are then generated as datasets/synthetic/samples_train_libsvm.txt and datasets/synthetic/samples_test_libsvm.txt in the libsvm format and as datasets/synthetic/samples_train_DOK.txt and datasets/synthetic/samples_test_DOK.txt in the DOK format. 


 *   (1.2).  Our program reads libsvm format.


* (2). Real Experiments:

> ./bin/spark-submit --class edu.uci.eecs.spectralLDA.SpectralLDA --master local[2] --deploy-mode client <PROJECT-PATH>/SpectralLDA-TensorSpark/SpectralLDA-Tensor/target/scala-2.10/SpectralLDA-Tensor-assembly-1.0.jar <PROJECT-PATH>/SpectralLDA-TensorSpark/src/main/resources/Data/datasets/enron_email/corpus.txt

*   (2.1).  Our program takes raw text (NOTE: Each text file line should hold 1 document). 

 

* Dependencies

* * You should install sbt before running this program.
* * See [build.sbt]for the dependencies.


### References ###
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html




### Who do I talk to? ###

* Repo owner or admin: Furong Huang 


* Contact: furongh@uci.edu