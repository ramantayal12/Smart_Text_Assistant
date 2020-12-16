import tkinter as tk
from functools import partial

import pytesseract
from PIL import Image
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.ensemble import RandomForestClassifier as RFC

def show_text(text):
    win = tk.Tk()
    win.title(' Smart_Text_Assistant ')
    label = tk.Label(win, text=text, width=100, height=10)
    label.pack()
    win.mainloop()


def text_analysis():
    #show_text('Please Enter Input in Command Line')
    train_data = pd.read_csv('train_analysis_text.csv')
    category = train_data['label']
    features = ('text')
    x_train = train_data[features]

    cv = CV(min_df=0.01, max_df=0.9, ngram_range=(1, 3))
    x_train_features = cv.fit_transform(x_train)

    algo = RFC(max_depth=38)
    algo.fit(x_train_features, category)

    #    filename = 'finalized_text_analysis_model.sav'
    #    model = pickle.load(open(filename, 'rb'))

    text_to_analyze = input('Please Text')
    text_to_analyze = cv.transform([text_to_analyze])
    ans = algo.predict(text_to_analyze)
    if ans == 0:
        print('The Sentiment of text is : Negative')
        show_text('The Sentiment of text is : Negative')
    else:
        print('The Sentiment of text is : Positive')
        show_text('The Sentiment of text is : Positive')


def show_predicted_text(text):
    win = tk.Tk()
    win.title(' Smart_Text_Assistant ')

    label1 = tk.Label(win, text="The Predicted text is : ", bd=10, width=100, height=10)
    label1.pack()
    label2 = tk.Label(win, text=text)
    label2.pack()
    win.mainloop()


def text_creation(seed_text, word_length):
    with open('text_creation_model', 'r') as json_file:
        json_saved_model = json_file.read()
    # load the model architecture
    model_j = tf.keras.models.model_from_json(json_saved_model)
    # model_j = model_from_json('text_creation_model')
    model_j.load_weights('text_creation_model_weights.h5')

    text = open('book.txt', 'r').read()
    text = text.lower()
    sentences = text.split('\n')
    tokenizer = Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(sentences)

    for _ in range(word_length):
        sequence = tokenizer.texts_to_sequences([seed_text])
        padded = pad_sequences(sequence, maxlen=19)
        predicted = model_j.predict_classes(padded, verbose=0)
        out_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                out_word += word
                break
        seed_text += ' ' + out_word
    return seed_text


def but2():
    #show_text('Please Enter Input in Command Line')
    win = tk.Tk()
    win.title(' Smart_Text_Assistant ')

    label1 = tk.Label(win, text=" Please Enter the Text to be predicted : ",
                      bd=10, underline=10, height=10, width=100)
    label1.pack()
    text = tk.StringVar()
    entry1 = tk.Entry(win, textvariable=text)
    entry1.pack()

    seed_text = input('Please Enter Text ')
    word_len = int(input('Please Enter Number of Words to predict after provided Sentence'))

    predicted_text = text_creation(seed_text, word_len)

    button1 = tk.Button(win, text="Predict", height=10, width=100,
                        command=partial(show_predicted_text, predicted_text))
    button1.pack()

    win.mainloop()


def show_extracted_text():
    show_text('Please Enter Input in Project Folder')
    win = tk.Tk()
    win.title(' Smart_Text_Assistant ')

    label1 = tk.Label(win, text="The Extracted text is : ", bd=10, underline=10)
    label1.pack()
    extracted_text = (pytesseract.image_to_string(Image.open('test1.png')))
    label2 = tk.Label(win, text=extracted_text)
    label2.pack()
    win.mainloop()


def but1():
    win = tk.Tk()
    win.title(' Smart_Text_Assistant ')
    button1 = tk.Button(win, text=" Provide Image in main folder Then Click this button",
                        height=10, width=100, command=show_extracted_text)
    button1.pack()
    win.mainloop()


def main():

    top = tk.Tk()

    top.title(' Smart_Text_Assistant ')

    # Implementing Buttons
    button1 = tk.Button(top, text="1. Convert Image to Text  ", height=10, width=100, command=but1)
    button1.pack()

    button2 = tk.Button(top, text="2. Predict the Text using AI ", height=10, width=100, command=but2)
    button2.pack()

    button3 = tk.Button(top, text="3. Classify the Text-Sentiment using AI ",
                        height=10, width=100, command=text_analysis)
    button3.pack()

    button4 = tk.Button(top, text="4.  Exit the Application", height=10, width=100, command=top.destroy)
    button4.pack()

    top.mainloop()


if __name__ == '__main__':
    main()
