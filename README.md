# Accurate Arithmetic Methods in Real Floating-Point Arithmetic
## Description
Welcome to the GitHub Repository for my Penn State University Schreyer Honors Thesis on compensated arithmetic. The thesis abstract is outlined below, and I will provide a hyperlink for the open access copy in a future update.

## Authors
* Nathan Murarik
* Advisor: Dr. Thomas R. Cameron 
  * Assistant Professor at Penn State Erie, The Behrend College
  * Email: [trc5475@psu.edu](mailto:trc5475@psu.edu)
  
## Abstract
Compensated arithmetic is a summation technique designed to filter the error generated by a floating-point computation. The filtered error is then used to accurately estimate the computation's exact value. In this thesis, we use compensated arithmetic to construct accurate addition, multiplication, and division algorithms in real floating-point arithmetic. We demonstrate their accuracy by proving each algorithm's output is as accurate as if computed in k-fold precision and stored in k-parts. Moreover, we use the derived forward-error bounds to perform backward error-analysis on the dot product and on backward-substitution for upper-triangular systems. We further explore matrix applications by augmenting Gaussian Elimination with Partial Pivoting and iterative refinement with our k-parts arithmetic algorithms. We supplement our theoretical error bounds with numerical experiments on both ill-conditioned dot products and ill-conditioned linear systems.

## How to Use
I implement our methods in C on 1.7 GHz Quad-Core Intel Core i7 processor with 16 GB of memory. I use the Multi-Precision Floating-Point Reliable Library (MPFR) and Linear Algebra Package (LAPACKE) to conduct numerical experiments on ill-conditioned dot products and linear systems. Ensure both of these dependencies are available on your system and edit the CINC and CLIB portion of make.inc.

Now, go to the C folder your terminal and type the command ``make kpart_test`` to complie the project. You can then run ``./kpart_test [program] [vector/matrix size] [sample size] [min k] [max k]`` to execute a test. 
