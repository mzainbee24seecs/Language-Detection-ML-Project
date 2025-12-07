import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# ---------------------------
# 1. Load the dataset
# ---------------------------
data = pd.read_csv("data/language_dataset_50.csv")
texts = data['text'].values
labels = data['language'].values

# ---------------------------
# 2. Encode labels
# ---------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# ---------------------------
# 3. Tokenize text
# ---------------------------
max_words = 5000
max_len = 20  # max words per sentence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len, padding='post')

# ---------------------------
# 4. Split train/test
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Build the model
# ---------------------------
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ---------------------------
# 6. Train the model
# ---------------------------
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# ---------------------------
# 7. Evaluate the model
# ---------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")
