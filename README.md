# TensorFlow.js

## Tutorial: Making Predictions from 2D Data
### URL
https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html

### Notes
* Dense layers come with a bias term by default, so we do not need to set useBias to true, we will omit from further calls to tf.layers.dense.
* In this example, because the hidden layer has 1 unit, we don't actually need to add the final output layer above (i.e. we could use hidden layer as the output layer). However, defining a separate output layer allows us to modify the number of units in the hidden layer while keeping the one-to-one mapping of input and output.

### Best Practices
* You should always shuffle your data before handing it to the training algorithms in TensorFlow.js.
* You should always consider normalizing your data before training. Some datasets can be learned without normalization, but normalizing your data will often eliminate a whole class of problems that would prevent effective learning. You can normalize your data before turning it into tensors. We do it afterwards because we can take advantage of vectorization in TensorFlow.js to do the min-max scaling operations without writing any explicit for loops.

### Main Takaways
The steps in training a machine learning model include:

* Formulate your task:
  * Is it a regression problem or a classification one?
  * Can this be done with supervised learning or unsupervised learning?
  * What is the shape of the input data? What should the output data look like?
* Prepare your data:
  * Clean your data and manually inspect it for patterns when possible
  * Shuffle your data before using it for training
  * Normalize your data into a reasonable range for the neural network. Usually 0-1 or -1-1 are good ranges for numerical data.
  * Convert your data into tensors
* Build and run your model:
  * Define your model using **tf.sequential** or **tf.model** then add layers to it using **tf.layers.***
  * Choose an optimizer ( adam is usually a good one), and parameters like batch size and number of epochs.
* Choose an appropriate loss function for your problem, and an accuracy metric to help your evaluate progress. meanSquaredError is a common loss function for regression problems.
  * Monitor training to see whether the loss is going down
* Evaluate your model
  * Choose an evaluation metric for your model that you can monitor while training. Once it's trained, try making some test predictions to get a sense of prediction quality.

---

## Tutorial: Handwritten Digit Recognition with CNNs
### URL
https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html

### Goal
To train a model that will take one image and learn to predict a score for each of the possible 10 classes that image may belong to (the digits 0-9).

Each image is 28px wide 28px high and has a 1 color channel as it is a grayscale image. So the shape of each image is [28, 28, 1].

### Notes
* There are no weights in a flatten layer. It just unrolls its inputs into a long array.
* When using the Layers API loss and accuracy is computed on each batch and epoch.
* _trainDataSize_ is set to 5500 and testDataSize to 1000 to make it faster to experiment with. Once you've got this tutorial running feel free to increase that to 55000 and 10000 respectively. It will take a bit longer to train but should still work in the browser on many machines.
* We do not use any probability threshold here. We take the highest value even if it is relatively low. An interesting extension to this project would be to set some required minimum probability and indicate ‘no digit found' if no class meets this classification threshold.
* doPredictions shows how you can generally make predictions once your model is trained. However with new data you wouldn't have any existing labels

### Main Takeaways
* Predicting categories for input data is called a classification task.
* Classification tasks require an appropriate data representation for the labels
  * Common representations of labels include one-hot encoding of categories
* Prepare your data:
  * It is useful to keep some data aside that the model never sees during training that you can use to evaluate the model. This is called the validation set.
* Build and run your model:
  * Convolutional models have been shown to perform well on image tasks.
  * Classification problems usually use categorical cross entropy for their loss functions.
  * Monitor training to see whether the loss is going down and accuracy is going up.
* Evaluate your model
  * Decide on some way to evaluate your model once it's trained to see how well it is doing on the initial problem you wanted to solve.
  * Per class accuracy and confusion matrices can give you a finer breakdown of model performance than just overall accuracy.
