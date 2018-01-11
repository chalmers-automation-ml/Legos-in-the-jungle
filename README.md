# Legos-in-the-jungle

We've put together an example generator called "Legos in the jungle", where a Lego of roughly size 40x20 is random (uniformly) rotated and placed in a 100x100 jungle image (subsampled from a larger one), all in black and white.

The code for this first challenge is found in /Challenge-1/. By running generateData.py you create the two data sets with 5000 samples (by default), the first 1000 which are designated validation data.

There are two data-sets, in the first there is a probability of 0.5 that a Lego is actually placed in the jungle and the objective is to classify the image as containing or not containing a Lego, in the second there is always a Lego and the task is to estimate the position and rotation.

We've modified and repaired the MNIST neural net example used in "Keras Tutorial: The Ultimate Beginner's Guide to Deep Learning in Python" to fit with our data (exact same network architecture but for input and output). On the classification task we get 0.9304 accuracy, and when we do the estimation 0.9147. For this example let us define accuracy as the average validation accuracy over 10 epochs.

We challenge you to improve this number, by ANY means! Be creative! If you can beat the current best, add your names and scores to the leader board (keep the history) and short note on what you did to succeed, also post your solution to /Challenge-1/Solutions.


# Leader board 

## Challenge 1

### Classification

0.9877 Mattias (using a reduced version of the VGGNet)

0.9304 Oskar and Kristofer (using Keras Tutorial: The Ultimate Beginner's Guide....)

0.xxx New leaders (using tricks y and z)

### Estimation

0.9793 Mattias (using a reduced version of the VGGNet with quadratic penalty)

0.9147 Oskar and Kristofer (using Classification result + quadratic penalty)

0.xxx New leaders (using tricks y and z)

