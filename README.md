# ML-Learning-Algorithms
This repo contains code for some basic supervised learning algorithms (linear regression, logistic regression, gaussian discriminant analysis)

## Linear Regression

To perform supervised learning, we must decide how we’re going to represent functions/hypotheses h in a computer. As an initial choice, lets say we decide to approximate y as a linear function of x:

<math>h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> + &theta;<sub>1</sub>x<sub>1</sub> +  &theta;<sub>2</sub>x<sub>2</sub></math>

h<sub>&theta;</sub>(x) = &theta;<sup>T</sup>x

Here, the θi’s are the parameters (also called weights) parameterizing the space of linear functions mapping from X to Y. When there is no risk of confusion, we will drop the θ subscript in hθ(x), and write it more simply as h(x). 
 
 
**The loss function is** 

<math>J(&theta;) = 1/2 &sum; (h<sub>&theta;</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup> </math>

where i ranges from  1 to the number of training examples.

**The gradient descent step:**
We want to choose θ so as to minimize J(θ). To do so, lets use a search algorithm that starts with some “initial guess” for θ, and that repeatedly changes θ to make J(θ) smaller, until hopefully we converge to a value of θ that minimizes J(θ). Speciﬁcally, lets consider the gradient descent algorithm, which starts with some initial θ, and repeatedly performs the update: 

<math>&theta;<sub>j</sub> = &theta;<sub>j</sub> - &eta;.(&delta;/&delta;&theta;<sub>j</sub>).J(&theta;)</math>

The animation for the gradient descent of the parameters is done in the code.

    1. The python code for linear regression is named linear_regression.py
    2. It works on the sample data provided (linearX.csv is the x-axis or the training example set, linearY.csv is the prediction set)
    3. The points are normalised before training.
    4. the function for gradient descent is made and convergence criterion is also set.
    5. The meta values like convergence criterion can be altered in the code. Learning rate parameter is user input.
    6. The code is provided with functions that output animation of the learing step and the update of model parameters wrt learning rate.

To run the code use the following command:
```
python linear_regression.py linearX.csv linearY.csv learning_rate time_delay
```
here learning_rate is input by the user which determines the learning rate of the updation of parameters in training step.
time_delay is the integer which determines the sampling rate in the animation of the convergence of parameters updation.


## Weighted Linear Regression

In the weighted linear regression, the training step is repeated for the prediction of each new example, as those points which are nearer to the given
test points are given more weight in training step.

The normal equation is directly used to make prediction Y of individual points in the case of weighted linear regression.

The normal equation for simple linear regression is:           
𝜽 = (𝑿<sup>𝑻</sup>𝑿)<sup>−𝟏</sup>(𝑿<sup>𝑻</sup>𝒀) 

The normal equation for weighted linear regression becomes:           
𝜽 = (𝑿<sup>𝑻</sup>W𝑿)<sup>−𝟏</sup>(𝑿<sup>𝑻</sup>W𝒀) 

where, **W** is the weight matrix calculated in the provided code, using the formula:

<math>w<sup>(i)</sup>  = exp(-(x<sup>(i)</sup> - x)<sup>2</sup> / &tau;<sup>2</sup>)</math>

here x is the given test point and x(i) is the single training point.
each w<sup>(i)</sup> is the diagonal element of the weight matrix **W**

where, <math>&tau;</math> is called the bandwidth parameter and controls the rate at which <math>w<sup>(i)</sup></math> falls with distance from x.

The prediction by the code can be checked with different values of the parameter <math>&tau;</math>. 

graphical analysis is provided in the code for better understanding.

To run the code use the following command:
```
python weighted_linear_regression.py weightedX.csv weightedY.csv tau
```

## Logistic Regression

The learning rate and the convergence threshold can be changed in the code. 

To run the code use the following command:
```
python logistic_regression.py logisticX.csv logisticY.csv
```
## Gaussian Descriminant Analysis

there are two different codes, one with linear separator and other with a quadratic separator. 
The result from both of them is provided in the attached report.

To run the code use the following command:
```
python gaussian_descriminanat_analysis.py logisticX.csv logisticY.csv
or
python gda2.py logisticX.csv logisticY.csv
```
