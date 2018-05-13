#
#Model 1: One-Word-in, one-word-out-sequences

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import plot_model


# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = array(encoded)
        # predict a word in the vocabulary
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text, result = out_word, result + ' ' + out_word
    return result


# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i - 1:i + 1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# split into X and y elements
sequences = array(sequences)
X, y = sequences[:, 0], sequences[:, 1]
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)
plot_model(model, to_file='Model-1.jpg', show_layer_names=True, show_shapes=True)
# evaluate
print(generate_seq(model, tokenizer, 'Jack', 6))

#Model 2: Line-by-Line Sequence

#Create line based sequence
sequences=list()
for line in data.split('\n'):
    encoded=tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence=encoded[:i+1]
        sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))

#Padding the sequences, which will involve to find the longest sequence
#Thene using that as the length by which to pad-out all the other sequences

#Padd input sequences
max_length=max([len(seq) for seq in sequence])
sequences=pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' %max_length)

#Split the sequence into input and output
sequences= array(sequences)
X, y=sequences[:, :-1], sequences[:, -1]
y=to_categorical(y, num_classes=vocab_size)

#Define the model
model=Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(mode.summary())

#Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X, y, epochs=500, verbose=2)

#Generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text= seed_text

    #generate a fixed number of words
    for _ in range(n_words):
        #encode the text as integer
        encoded=tokenizer.texts_to_sequences([in_text])[0]

        #pre-pad sequences to a fixed length
        encoded=pad_sequences([encoded], maxlen=max_length, padding='pre')

        #Predict probabilities for each word
        yhat= model.predict_classes(encoded, verbose=0)

        #map predicted word index to word
        out_word= ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        #append to input
        in_text+= ' ' + out_word
    return in_text