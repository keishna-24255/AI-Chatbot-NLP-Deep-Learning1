import json
import pickle
import random
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


# -----------------------------
# 1. Initialize NLTK
# -----------------------------

lemmatizer = WordNetLemmatizer()

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")


# -----------------------------
# 2. Load chatbot data
# -----------------------------

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)


words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]


# -----------------------------
# 3. Process the intents
# -----------------------------

for intent in intents["intents"]:

    for pattern in intent["patterns"]:

        # Tokenize each sentence
        word_list = nltk.word_tokenize(pattern)

        words.extend(word_list)

        # Store pattern and tag
        documents.append((word_list, intent["tag"]))

        # Add tag to classes
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# -----------------------------
# 4. Lemmatize words
# -----------------------------

words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_words
]

words = sorted(set(words))
classes = sorted(set(classes))


print("Words:", len(words))
print("Classes:", len(classes))
print("Documents:", len(documents))


# -----------------------------
# 5. Save words and classes
# -----------------------------

with open("words.pkl", "wb") as file:
    pickle.dump(words, file)

with open("classes.pkl", "wb") as file:
    pickle.dump(classes, file)


# -----------------------------
# 6. Create training data
# -----------------------------

training = []

output_empty = [0] * len(classes)


for document in documents:

    bag = []

    pattern_words = document[0]

    # Lemmatize the pattern words
    pattern_words = [
        lemmatizer.lemmatize(word.lower())
        for word in pattern_words
    ]

    # Create Bag of Words
    for word in words:

        if word in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # Create output class
    output_row = list(output_empty)

    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])


# -----------------------------
# 7. Shuffle training data
# -----------------------------

random.shuffle(training)

training = np.array(training, dtype=object)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])


print("Training data created.")
print("Input shape:", train_x.shape)
print("Output shape:", train_y.shape)


# -----------------------------
# 8. Build Neural Network
# -----------------------------

model = Sequential()

model.add(
    Dense(
        128,
        input_shape=(len(train_x[0]),),
        activation="relu"
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        len(train_y[0]),
        activation="softmax"
    )
)


# -----------------------------
# 9. Compile model
# -----------------------------

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)


# -----------------------------
# 10. Train model
# -----------------------------

print("\nTraining chatbot model...\n")

history = model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)


# -----------------------------
# 11. Save trained model
# -----------------------------

model.save("chatbot_model.keras")


print("\n--------------------------------")
print("Chatbot training completed!")
print("--------------------------------")
print("Model saved as chatbot_model.keras")
print("Words saved as words.pkl")
print("Classes saved as classes.pkl")