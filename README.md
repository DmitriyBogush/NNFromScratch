# Neural Network From Scratch
Creating a Neural Network from scratch using Python.  

**Features:**
- Arbitrary Depth 
- Minibatching
- Classification
- Regression

**Allowed libraries:**  
- Numpy
- Argparse
- Math

**Usage:**  
```NeuralNetwork.py -v -train_feat FILENAME -train_target FILENAME -dev_feat FILENAME -dev_target FILENAME -epochs # -learnrate # -nunits # -type [c,r] -hidden_act [sig,relu,tanh] -init_range # -num_classes # -mb # -nlayers # ``` 

```-v```: verbose mode, minibatch and det set performance printed at each update.   
```-train_feat```: name of training set feature file.  
```-train_target```: name of training set target file.   
```-dev_feat```: name of development set feature file.  
```-dev_target```: name of developement set target file.  
```-epochs```: number of epochs.  
```-learnrate```: step size.   
```-nunits```: dimension of hidden layer.  
```-type```: c = classification | r = regression.   
```-hidden_act```: hidden layer activation function | sig = logistic (sigmoid) | relu = rectified linear activation (ReLU) | tanh = hyperbolic tangent (Tanh).  
```-init-range```: weights, including bias vector.  
```num-classes```: Classification: number of classes. | Regression: dimensions of output vector.  
```-mb```: size of each minibatch.  
```-nlayers```: numbers of hidden layers. 
