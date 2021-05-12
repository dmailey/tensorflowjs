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

---

## Tutorial: Handwritten Digit Recognition with CNNs
