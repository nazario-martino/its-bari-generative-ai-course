import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import sys

# 1. Caricamento e preparazione del dataset (esempio con il testo di Shakespeare)
# Puoi usare un testo diverso (ad esempio poesie di Emily Dickinson) o scaricare il testo da Project Gutenberg

text = open('shakespeare.txt', 'r').read().lower()  # Carica un file di testo
print(f"Numero di caratteri nel corpus: {len(text)}")

# Creazione del vocabolario
chars = sorted(list(set(text)))  # Lista di caratteri unici
print(f"Numero di caratteri unici: {len(chars)}")

char_to_idx = {char: idx for idx, char in enumerate(chars)}  # Mappa carattere -> indice
idx_to_char = {idx: char for idx, char in enumerate(chars)}  # Mappa indice -> carattere

# 2. Preprocessing del testo
seq_length = 40  # Lunghezza della sequenza di input
sequences = []
next_chars = []

# Creazione delle sequenze di input e dei target
for i in range(0, len(text) - seq_length, 1):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

# Conversione in numeri
X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, seq in enumerate(sequences):
    for j, char in enumerate(seq):
        X[i, j, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# 3. Creazione del modello LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# 4. Compilazione del modello
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 5. Addestramento del modello
model.fit(X, y, batch_size=128, epochs=20)

# 6. Funzione per generare testo
def generate_text(seed, length=200, temperature=1.0):
    print(f"Generando testo a partire da: '{seed}'")
    generated = seed
    seed = seed.lower()
    for _ in range(length):
        x_pred = np.zeros((1, len(seed), len(chars)))
        for i, char in enumerate(seed):
            x_pred[0, i, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        # Applicazione della temperatura
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # Selezione del prossimo carattere
        next_index = np.random.choice(len(chars), p=preds)
        next_char = idx_to_char[next_index]

        # Aggiunta del carattere generato alla sequenza
        generated += next_char
        seed = seed[1:] + next_char

    return generated

# 7. Test del modello con un seed
seed_text = "shall i compare thee"
generated_text = generate_text(seed_text, length=300, temperature=1.0)
print(generated_text)

