import numpy as np
import re
from tensorflow.keras import models, layers
import random

# Funzione per caricare il testo da un file .txt
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Converte tutto in minuscolo
    return text

# Esempio di utilizzo: Carica il testo dal file
# Assicurati che il percorso del file sia corretto
file_path = 'path_to_your_text_file.txt'  # Inserisci il percorso del file di testo
testo_training = load_text_from_file(file_path)

# Preprocessing del testo
def preprocess_text(text):
    """Pulisce e prepara il testo"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Rimuovi caratteri speciali
    words = text.split()
    return words

words = preprocess_text(testo_training)
print(f"Corpus: {len(words)} parole")
print(f"Prime 5 parole: {words[:5]}")

# Creazione vocabolario
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# Funzione per creare le sequenze di training
def create_training_sequences(words, seq_length):
    sequences = []
    targets = []
    for i in range(len(words) - seq_length):
        seq = words[i:i+seq_length]
        target = words[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return sequences, targets

# Funzione per convertire le parole in indici
def words_to_indices(words, word_to_ix):
    return [word_to_ix[word] for word in words]

# Funzione per creare il vettore one-hot
def create_one_hot(index, vocab_size):
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot

# Creazione delle sequenze di training
seq_length = 3  # Lunghezza della sequenza di input
sequences, targets = create_training_sequences(words, seq_length)

# Conversione in formato numerico
X = []
y = []

for seq, target in zip(sequences, targets):
    seq_indices = words_to_indices(seq, word_to_ix)
    seq_one_hot = [create_one_hot(idx, vocab_size) for idx in seq_indices]
    target_idx = word_to_ix[target]
    target_one_hot = create_one_hot(target_idx, vocab_size)
    
    X.append(seq_one_hot)
    y.append(target_one_hot)

X = np.array(X)
y = np.array(y)

# Costruzione del modello RNN
model = models.Sequential([
    layers.SimpleRNN(50, input_shape=(seq_length, vocab_size), activation='tanh', return_sequences=False),
    layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(X, y, epochs=50, batch_size=8)

# Funzione per generare il testo
def generate_text(model, seed_text, num_words=10, temperature=1.0):
    generated_text = seed_text[:]
    
    for _ in range(num_words):
        sequence = [word_to_ix[word] for word in seed_text]
        sequence_one_hot = [create_one_hot(idx, vocab_size) for idx in sequence]
        sequence_one_hot = np.array(sequence_one_hot).reshape(1, len(seed_text), vocab_size)

        # Predizione della parola successiva
        preds = model.predict(sequence_one_hot, verbose=0)[0]
        
        # Applicazione della temperatura
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # Selezione della parola successiva in base alla probabilità
        next_idx = np.random.choice(len(vocab), p=preds)
        next_word = ix_to_word[next_idx]
        
        # Aggiunta della parola generata alla sequenza
        generated_text.append(next_word)
        seed_text = seed_text[1:] + [next_word]
    
    return generated_text

# Generazione di una poesia
seed_text = ["la", "luna", "splende"]
generated_text = generate_text(model, seed_text, num_words=10, temperature=1.0)
print(f"Testo generato: {' '.join(generated_text)}")
