import json
import pickle
import random
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# -----------------------------------
# 1. Initialize
# -----------------------------------

lemmatizer = WordNetLemmatizer()

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


# -----------------------------------
# 2. Load chatbot files
# -----------------------------------

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

with open("words.pkl", "rb") as file:
    words = pickle.load(file)

with open("classes.pkl", "rb") as file:
    classes = pickle.load(file)

model = load_model("chatbot_model.keras")


# -----------------------------------
# 3. Clean the sentence
# -----------------------------------

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]

    return sentence_words


# -----------------------------------
# 4. Create Bag of Words
# -----------------------------------

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for sentence_word in sentence_words:

        for index, word in enumerate(words):

            if word == sentence_word:
                bag[index] = 1

    return np.array(bag)


# -----------------------------------
# 5. Predict the intent
# -----------------------------------

def predict_class(sentence):

    bow = bag_of_words(sentence)

    prediction = model.predict(
        np.array([bow]),
        verbose=0
    )[0]

    ERROR_THRESHOLD = 0.25

    results = [
        [index, probability]
        for index, probability in enumerate(prediction)
        if probability > ERROR_THRESHOLD
    ]

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "intent": classes[result[0]],
            "probability": str(result[1])
        }
        for result in results
    ]


# -----------------------------------
# 6. Get response
# -----------------------------------

def get_response(intents_list):

    if not intents_list:
        return "Sorry, I don't understand that."

    tag = intents_list[0]["intent"]

    for intent in intents["intents"]:

        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't know the answer."


# -----------------------------------
# 7. Start Chatbot
# -----------------------------------

print("\n===================================")
print("       AI CHATBOT")
print("===================================")
print("Type 'quit' to exit.")
print("")


while True:

    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! Have a great day!")
        break

    predicted_intent = predict_class(user_input)

    response = get_response(predicted_intent)

    print("Bot:", response)