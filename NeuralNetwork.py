import argparse
import numpy as np
import math 
import sys 

parser = argparse.ArgumentParser() 

# Adding the command line arguments 
parser.add_argument('-v', action = "store_true")
parser.add_argument('-train_feat', metavar = 'TRAIN_FEAT_FN', required = True)
parser.add_argument('-train_target', metavar = 'TRAIN_TARGET_FN', required = True)
parser.add_argument('-dev_feat', metavar = 'DEV_FEAT_FN', required = True)
parser.add_argument('-dev_target', metavar = 'DEV_TARGET_FN', required = True)
parser.add_argument('-epochs', metavar = 'EPOCHS', required = True, type = int)
parser.add_argument('-learnrate', metavar = 'LEARNRATE', required = True, type = float)
parser.add_argument('-nunits', metavar = 'NUM_HIDDEN_UNITS', required = True, type = int)
parser.add_argument('-type', metavar = 'PROBLEM_MODE', required = True, type = str.lower, choices=['c', 'r'])
parser.add_argument('-hidden_act', metavar = 'HIDDEN_UNIT_ACTIVATION', required = True, type = str.lower, choices=['sig', 'relu', 'tanh'])
parser.add_argument('-init_range', metavar = 'INIT_RANGE', required = True, type = float)
parser.add_argument('-num_classes', metavar ='C', type = int)
parser.add_argument('-mb', metavar = 'MINIBATCH_SIZE', type = int)
parser.add_argument('-nlayers', metavar = 'NUM_HIDDEN_LAYERS', type = int)

args = parser.parse_args()

# Layer class 
class Layer: 
    def __init__(self):
        self.W     = None 
        self.b     = None
        self.a     = None
        self.gradW = None 
        self.gradb = None 
        self.delta = None

# Load the data from training set files. 
def loadTrainingSet():
    features = open(args.train_feat, 'r')
    targets = open(args.train_target, 'r')
    
    # Load in all y values
    ready = targets.read().splitlines()
    rowsy = [row.split() for row in ready]
    y = np.array(rowsy, float)
    
    # If classification -> one hot encode the y values 
    if args.type == 'c':
        rowhot = []
        for row in rowsy:
            hot = np.zeros(args.num_classes)
            hot[int(row[0])] = 1
            rowhot.append(hot)

        y = np.array(rowhot, float)  

    # Load in all x values 
    read = features.read().splitlines()
    rows = [row.split() for row in read]
    x = np.array(rows, float)
    D = np.shape(x)[1]

    features.close()
    targets.close()

    return y,x,D  

# Load the data from dev set files. 
def loadDevSet():
    features = open(args.dev_feat, 'r')
    targets = open(args.dev_target, 'r')
    
    # Load in all x values 
    read = features.read().splitlines()
    rows = [row.split() for row in read]
    x = np.array(rows, float)
    
    # Load in all y values
    ready = targets.read().splitlines()
    rowsy = [row.split() for row in ready]
    y = np.array(rowsy, float)
    
    # If classification -> one hot encode the y values 
    if args.type == 'c':
        rowhot = []
        for row in rowsy:
            hot = np.zeros(args.num_classes)
            hot[int(row[0])] = 1
            rowhot.append(hot)

        y = np.array(rowhot, float)  
  
    features.close()
    targets.close()

    return y,x 

# Given a pre-activation layer will apply a reLU activation function. 
def relu(preAct):
    return np.maximum(0, preAct)

# Given a pre-activation layer will apply a sigmoid activation function. 
def sigmoid(preAct):
    return 1 / (1 + np.exp(-preAct))

# Derivatives 
def sigmoidDerv(postAct):
    return np.multiply(postAct, np.ones(postAct.shape) - postAct)

def tanhDerv(postAct):
    return np.ones(postAct.shape) - (postAct * postAct)

def reluDerv(postAct):    
    result = np.where(postAct >= 0, 1, postAct)
    result = np.where(postAct < 0, 0, postAct)
    return result 

def softmax(preAct):
    # Calculate the exp
    minimized = preAct - np.max(preAct)
    exped = np.exp(minimized)
    
    # Normalize 
    result = exped / exped.sum(axis = 1)[:,None]
    return result

# Returns a list of layers with wieghts that have been randomly assigned. 
def init_weights(D):
    # Create a list of hidden layers +1 for output layer.
    layers = [] 

    # If number of layers is 0 -> Linear model 
    if args.nlayers == 0: 
        # First layer 
        W = np.random.uniform(-args.init_range, args.init_range, (D,args.nunits))
        b = np.random.uniform(-args.init_range, args.init_range, (1,args.nunits)) 
        newlayer = Layer()
        newlayer.W = W
        newlayer.b = b 
        layers.append(newlayer)

        # Add the output layer 
        W = np.random.uniform(-args.init_range, args.init_range, (args.nunits,args.num_classes))
        b = np.random.uniform(-args.init_range, args.init_range, (1,args.num_classes)) 
        outputlayer = Layer()
        outputlayer.W = W
        outputlayer.b = b 
        layers.append(outputlayer)
    else: 
        # Non linear model 
        for i in range(args.nlayers):
            if i == 0: 
                # First layer 
                W = np.random.uniform(-args.init_range, args.init_range, (D,args.nunits))
                b = np.random.uniform(-args.init_range, args.init_range, (1,args.nunits)) 
                newlayer = Layer()
                newlayer.W = W
                newlayer.b = b 
                layers.append(newlayer)
            else: 
                W = np.random.uniform(-args.init_range, args.init_range, (args.nunits,args.nunits))
                b = np.random.uniform(-args.init_range, args.init_range, (1,args.nunits)) 
                newlayer = Layer()
                newlayer.W = W
                newlayer.b = b 
                layers.append(newlayer)
        
        if args.nlayers != 0: 
            # Add the output layer 
            W = np.random.uniform(-args.init_range, args.init_range, (args.nunits,args.num_classes))
            b = np.random.uniform(-args.init_range, args.init_range, (1,args.num_classes)) 
            outputlayer = Layer()
            outputlayer.W = W
            outputlayer.b = b 
            layers.append(outputlayer)
        else:
            W = np.random.uniform(-args.init_range, args.init_range, (D,args.num_classes))
            b = np.random.uniform(-args.init_range, args.init_range, (1,args.num_classes)) 
            outputlayer = Layer()
            outputlayer.W = W
            outputlayer.b = b 
            layers.append(outputlayer)

    return layers

def forward(miniX, layers):

    # For every layer 
    for i in range(len(layers)): 
        curLayer = layers[i] 
        
        # Compute the preactivation
        # If first layer get raw data otherwise previous guys a 
        if i == 0: 
            preAct = np.add(np.matmul(miniX, curLayer.W),curLayer.b)
        else: 
            preAct = np.add(np.matmul(layers[i - 1].a, curLayer.W),curLayer.b)

        # Compute the postactivation
        # If an output layer: 
        if i == len(layers) - 1: 
            # Regression -> identity
            if args.type == 'r': 
                postAct = preAct 
                curLayer.a = postAct 
                curLayer.z = preAct   
                return postAct  
            else: 
                # Binary Classification -> sigmoid 
                if args.num_classes == 2:
                    postAct = sigmoid(preAct)
                    curLayer.a = postAct 
                    curLayer.z = preAct   
                    return postAct

                # Multi Class Classification -> softmax 
                if args.num_classes > 2: 
                    postAct = softmax(preAct)
                    curLayer.a = postAct 
                    curLayer.z = preAct   
                    return postAct
        else: 
            # A hidden layer 
            if args.hidden_act == 'tanh':
                postAct = np.tanh(preAct)
            if args.hidden_act == 'sig':
                postAct = sigmoid(preAct)
            if args.hidden_act == 'relu': 
                postAct = relu(preAct)
        
            curLayer.a = postAct 
            curLayer.z = preAct     
        

def computeLoss(a, miniY):
    if args.type == 'c':
        # Classification -> Accuracy metric
        count = 0 
        for i in range(len(a)):
            if np.argmax(a[i]) == np.argmax(miniY[i]):
                count += 1 
        return count / len(a)
    else: 
        # Regression -> Mean Squared error loss
        return np.square(np.subtract(miniY, a)).mean()

# Backprop algorithm will give us the gradients 
def backprop(layers, a, miniY, miniX):

    norm = miniX.shape[0]

    # Loop through the layers backwards
    for i in reversed(range(len(layers))): 
        currLayer = layers[i]

        if i == len(layers) - 1: 
            # Last layer 
            delta = np.subtract(a, miniY)
            currLayer.delta = delta 
            currLayer.gradW = np.matmul(layers[i-1].a.T, delta) * (1/norm)
            currLayer.gradb = np.matmul(np.ones(int(delta.shape[0])), delta) * (1/norm)
            
        else: 
            delta = layers[i + 1].delta

            if args.hidden_act == 'tanh':
                currLayer.delta = np.multiply(tanhDerv(currLayer.a), np.matmul(layers[i + 1].W, layers[i + 1].delta.T).T)
            if args.hidden_act == 'sig':
                currLayer.delta = np.multiply(sigmoidDerv(currLayer.a), np.matmul(layers[i + 1].W, layers[i + 1].delta.T).T)
            if args.hidden_act == 'relu':  
                currLayer.delta = np.multiply(reluDerv(currLayer.z), np.matmul(layers[i + 1].W, layers[i + 1].delta.T).T)

            if i-1 == -1:
                # Last in backprop -> first layer in model 
                currLayer.gradW = np.matmul(miniX.T, currLayer.delta) * (1/norm)
            else: 
                currLayer.gradW = np.matmul(layers[i-1].a.T, currLayer.delta) * (1/norm)
            currLayer.gradb = np.matmul(np.ones(int(currLayer.delta.shape[0])), currLayer.delta) * (1/norm)
            

# Given a list of layers will update all of the weights with the gradients times learn rate. 
def updateWeights(layers):
    for i in range(len(layers)):
        currLayer = layers[i]
        currLayer.W = currLayer.W - (args.learnrate * currLayer.gradW)
        currLayer.b = currLayer.b - (args.learnrate * currLayer.gradb)

# Main training loop for NN walks through all of the steps. 
def train (yTrain, xTrain, yDev, xDev, D): 

    # Create weights matrices and minibatch index array.
    layers    = init_weights(D) 
    minibatch = np.arange(0,len(xTrain))
    start     = 0 
    end       = 0 
    updates   = 0 
    times = [] 
 
    # For each epoch 
    for i in range(args.epochs):
        accur = []

        # Randomly shuffle the data 
        np.random.shuffle(minibatch)
        yTrainShuf = yTrain[minibatch]
        xTrainShuf = xTrain[minibatch]

        # If minibatch is 0 do full batch 
        if args.mb == 0: 
            batches = 1
        else: 
            batches = math.ceil(len(xTrain) / args.mb)
        
        # For each minibatch in the data 
        for j in range(batches):

            if (args.mb == 0): 
                start = 0 
                end   = len(xTrain) - 1 
                # If on the last one where no more minibatch will fit 
            elif(args.mb + (j * args.mb) > len(xTrain)):
                
                # Randomly reshuffle the data 
                np.random.shuffle(minibatch)
                yTrainShuf = yTrain[minibatch]
                xTrainShuf = xTrain[minibatch]
                
                # Set first interval 
                start = 0 
                end   = args.mb 
            else: 
                # Otherwise minibatch fits continue 
                start = (j * args.mb)
                end   = args.mb + (j * args.mb)

            # Do a feed forward pass
            a = forward(xTrainShuf[start : end], layers)

            # Compute the gradients with backprop 
            backprop(layers, a, yTrainShuf[start : end], xTrainShuf[start : end])

            # Update the weight matrices
            updateWeights(layers)
            updates += 1 

            # Get the minibatch score and add it to a list 
            minibatchScore = computeLoss(layers[len(layers)-1].a, yTrainShuf[start : end]) 
            accur.append(minibatchScore)

            # Verbose mode on: eval of dev and print update percentages
            if args.v == True:
                adev = forward(xDev, layers)
                devLoss = computeLoss(adev, yDev) 
                print("Update %06d" % updates, ": train=%.3f" % minibatchScore, "dev=%.3f" % (devLoss), file = sys.stderr)

        #Eval on dev and print 
        if args.v == False: 
            adev = forward(xDev, layers)
            devLoss = computeLoss(layers[len(layers)-1].a, yDev) 
            print("Epoch %03d" % (i + 1), ": train=%.3f" % (sum(accur) / len(accur)), "dev=%.3f" % (devLoss), file=sys.stderr)
        
    

# Driver code 
def main (): 
    yTrain,xTrain,D = loadTrainingSet()
    yDev, xDev = loadDevSet()     
    train(yTrain,xTrain,yDev,xDev,D)

# Call to main which will run the entire NN
main() 