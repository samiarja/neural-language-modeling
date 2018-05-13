# neural-language-modeling
# Predicting the next word in a sequence of words that appear right above your keyboard on your phone
<br>

![img](https://github.com/samiarja/neural-language-modeling/blob/master/word-based-nn-model/photo1.PNG)
<br>

Language modeling is a parts of **Natural Language Proccesing** that solve many workd problem in **Speech recognition** and **Machine Translation**, regardless, the improvment in this field, it is always beneficial to start from the basic of this model, to try to understand every aspects of this algorithm. Therefore, this is three types model for language modeling, which behave in different ways and give different desired output.
<br>

These NLP models are build under Keras framwork and all the neural network architechture are Sequential.
They are based on different approaches such as line by line sequence, two word input one word output.

**Model 1: One-Word-In, One-Word-Out Sequences**
<br>
> **word-in-out-seq.py**

**Model 2: Line-by-Line Sequence**

<br>

> **line-by-line-seq.py**

**Model 3: Two-Words-In, One-Word-Out Sequence**

<br>

> **two-word-in-one-out-seq.py**

<br>

A simple model that given a single word taken from some sentence tries predicting the word following it.
<br>

Using ***plot_model*** from Keras, I was able to visualize the neural network architecture.
<br>

![model1](https://github.com/samiarja/neural-language-modeling/blob/master/word-based-nn-model/Model-1.jpg)
<br>

This model is consisting of 4 layers:

* Embedding layer
* Hidden layer= Embedding
* Hidden layer= LSTM
* Output layer= Dense with softmax activation function

A brief explanation about Word-in, Word-out model
<br>

This model is seperated into major components:
* Encoding
* RNN
* Decoding
<br>

**Encoding**
<br>

It will start by using the one-hot-vectors, where the input will be encoded. Then taking the 1-hot vector representing the input word in the first NN layer, and multiply it by an input **Embedding** metrix of NxM shape. This embedding is a dense representation of the current input word.
Technically, while distance between every two consecutive words represented by a 1-hot vectors is the same, these dense representation on the other hand have the property that words that are close in meaning will have a representation that are closer to the embedding space.
<br>

**Recurrent Neural Network layers**
<br>

To improve performance of predicting the next word that actually make sense. It will work by adding memory to the system such as a LSTM(Long-short term memory)

**Decoding**
<br>

Which is the second component in the system, and which will always occur after the encoding step. Multiply the representation of the input word by output Embedding matrix of shape NxM. The resulting vector is passed through a **softmax** which is a probabilistic distribution to normalize its value to be between 0 and 1.
<br>




