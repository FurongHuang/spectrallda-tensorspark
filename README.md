# README #
* Quick summary: 
This code implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on Spark. 

* Version: 0.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### Run a Synthetic Experiment Toy Example
* install sbt
* open your terminal
* cd to SpectralLDA-Tensor
* type: "sbt" (or "sbt -mem <Customized Memory>", e.g., "sbt -mem 40960")"
)
* In sbt, type: "run src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt --synthetic 1"



### How do I get set up? ###


* Summary of set up
Main file is SpectralLDA-Tensor/src/main/scala-2.10/SpectralLDA.scala

* Configuration 

* (1). Synthetic Experiments:

> cd SpectralLDA-Tensor/

> sbt

> run src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt --synthetic 1


*    (1.1).  Data generation script in MATLAB is provided in the repository [here](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/data/SyntheticDataGenerator.m?at=master&fileviewer=file-view-default). One can play around with hyperparameters such as Sample Size, Vocabulary Size, Hidden Dimension, and How mixed the topics are.  The synthetic data for training are then generated as datasets/synthetic/samples_train_libsvm.txt and datasets/synthetic/samples_test_libsvm.txt in the libsvm format and as datasets/synthetic/samples_train_DOK.txt and datasets/synthetic/samples_test_DOK.txt in the DOK format. 


 *   (1.2).  Our program reads libsvm format.


* (2). Real Experiments:

> cd SpectralLDA-Tensor/

> sbt -mem 1024

> run <PATH_OF_YOUR_TEXT>

for example:

> run src/main/resources/Data/datasets/enron_email/corpus.txt

*   (2.1).  Our program takes raw text (NOTE: Each text file line should hold 1 document). 

 

* Dependencies

* * You should install sbt before running this program.
* * See [build.sbt]for the dependencies.


### References ###
* Project Page: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/community.html
* Paper: http://newport.eecs.uci.edu/anandkumar/pubs/Huang_etal_communities_GPU.pdf
* * F. Huang, U.N. Niranjan, M.U. Hakeem, A. Anandkumar, "Fast Detection of Overlapping Communities via Online Tensor Methods", JMLR 2014.




### Who do I talk to? ###

* Repo owner or admin: Furong Huang 


* Contact: furongh@uci.edu