# F1-maximization
A software package that trains an MLP by maximizing a soft approximation of the F1 squared.

One of the issues that we observe in applying MLP trained with MSE on natural language processing data is the unbalanced relation between precision and recall, which results in damage on the F1 measure. This issue is usually solved by manipulating the threshold; however, a larger gain on F1 can be obtained by manipulating all the parameters of the output layer of the MLP. Therefore, we developed this software package that fine-tunes the MLP output layer by maximizing a soft approximation of F1 squared.

The function train_NN_maxF1.m is used to train the MLP. This function receives as input the matrices of training data pairs (X, y) and the number of hidden neurons nneu and develops the trained matrixes of synaptic weights (W1, W2, b1, b2). Assuming a f-dimensional feature vector and a training set with N examples, the dimension of the input X has to be (f x N). The output has to be represented in one-hot style, i.e. if you have C classes, you have to have 1 for the position representing the target class and 0 for the other (C-1) classes. The output vector y has to have the dimension (C x N). After the training, the estimated output can be obtained using the function sim_NN.m, which returns the estimated output given the network parameters and the input matrix X. For details on this algorithm see Section 5.2 of the paper: Ludwig, Oswaldo, et al. "Deep Embedding for Spatial Role Labeling." arXiv preprint arXiv:1603.08474 (2016). This Matlab version is unconstrained, the objective function of Equation (33) has only the first term (i.e. C=0). 

In case of publication using this code, cite the paper:

@article{ludwig2016deep,
  title={Deep Embedding for Spatial Role Labeling},
  author={Ludwig, Oswaldo and Liu, Xiao and Kordjamshidi, Parisa and Moens, Marie-Francine},
  journal={arXiv preprint arXiv:1603.08474},
  year={2016}
}
