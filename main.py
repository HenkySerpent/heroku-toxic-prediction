import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = FastAPI()
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5

@app.get("/")
async def predict(X_test:np.array):
    model = tf.keras.models.load_model("savedModel/")
    X_test=pd.DataFrame(X_test)
    sentences = X_test[0]
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    p = model.predict(data)
    return p