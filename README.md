# README #
* Quick summary: 
This code implements a spectral (third order tensor decomposition) learning method for learning LDA topic model on         Spark. 

* Version: 0.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)



### How do I get set up? ###


* Summary of set up
For your convenience, the IDEA projected using [Intellij IDEA](https://www.jetbrains.com/idea/) is pushed in the repository [here](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/.idea/?at=master). 



* Configuration 
(1) Synthetic Experiments:


(a) Data generation script in MATLAB is provided in the repository [here](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/data/SyntheticDataGenerator.m?at=master&fileviewer=file-view-default). One can play around with hyperparameters such as Sample Size, Vocabulary Size, Hidden Dimension, and How mixed the topics are.  The synthetic data for training are then generated as datasets/synthetic/samples_train_libsvm.txt and datasets/synthetic/samples_test_libsvm.txt in the libsvm format and as datasets/synthetic/samples_train_DOK.txt and datasets/synthetic/samples_test_DOK.txt in the DOK format. 


(b) Our program reads libsvm format.


(c) One should remove the stopwords argument in the [configuration](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/src/main/scala/LDATensorDecomposition/SpectralLDA.scala?at=master&fileviewer=file-view-default) line 61 for synthetic experiments.


(2) Real Experiments:


(a) Data should be in the libsvm format. Our program takes raw text (document per row). 


   For example


===========


0 Apple:3 Fruit:4 Orange:1 ...


...


===========


(b) Change the synthetic argument in line 58 of the [configuration](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/src/main/scala/LDATensorDecomposition/SpectralLDA.scala?at=master&fileviewer=file-view-default) to 0.
 


* Dependencies


* * See [Code/tensorfac/build.sbt](https://bitbucket.org/furongh/spectral-lda/src/b5be6b9e2a45b824bbc60a0bb927eff6030f4256/Code/tensorfac/build.sbt?at=master&fileviewer=file-view-default) for the dependencies.


### References ###
* Project Page: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/community.html
* Paper: http://newport.eecs.uci.edu/anandkumar/pubs/Huang_etal_communities_GPU.pdf
* * F. Huang, U.N. Niranjan, M.U. Hakeem, A. Anandkumar, "Fast Detection of Overlapping Communities via Online Tensor Methods", JMLR 2014.


### Who do I talk to? ###

* Repo owner or admin: Furong Huang 


* Contact: furongh@uci.edu