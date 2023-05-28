import numpy as np
import pickle
import streamlit as st
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


#Lemmatizer and tfidf object
wordnet_lem = WordNetLemmatizer()


#Loading the saved model
MNB = pickle.load(open('E:/STUDY/3_2/CSE 3200/Sentiment Analysis/NLTK/MNB.sav', 'rb'))
MNB_tfidf = pickle.load(open('E:/STUDY/3_2/CSE 3200/Sentiment Analysis/NLTK/MNB_tfidf.sav', 'rb'))

MNB2 = pickle.load(open('E:/STUDY/3_2/CSE 3200/Sentiment Analysis/NLTK/MNB2.sav', 'rb'))
MNB2_CV2grams = pickle.load(open('E:/STUDY/3_2/CSE 3200/Sentiment Analysis/NLTK/MNB2_CV2grams.sav', 'rb'))

#For cleaning the input string
def cleaning(text):
    text = text.lower()  # converting to lowercase


    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text) 



    # removing short form: 
    
    text=re.sub("isn't",'is not',text)
    text=re.sub("aren't",'are not',text)
    text=re.sub("he's",'he is',text)
    text=re.sub("wasn't",'was not',text)
    text=re.sub("there's",'there is',text)
    text=re.sub("couldn't",'could not',text)
    text=re.sub("won't",'will not',text)
    text=re.sub("they're",'they are',text)
    text=re.sub("she's",'she is',text)
    text=re.sub("There's",'there is',text)
    text=re.sub("wouldn't",'would not',text)
    text=re.sub("haven't",'have not',text)
    text=re.sub("That's",'That is',text)
    text=re.sub("you've",'you have',text)
    text=re.sub("He's",'He is',text)
    text=re.sub("what's",'what is',text)
    text=re.sub("weren't",'were not',text)
    text=re.sub("we're",'we are',text)
    text=re.sub("hasn't",'has not',text)
    text=re.sub("you'd",'you would',text)
    text=re.sub("shouldn't",'should not',text)
    text=re.sub("let's",'let us',text)
    text=re.sub("they've",'they have',text)
    text=re.sub("You'll",'You will',text)
    text=re.sub("i'm",'i am',text)
    text=re.sub("we've",'we have',text)
    text=re.sub("it's",'it is',text)
    text=re.sub("don't",'do not',text)
    text=re.sub("that´s",'that is',text)
    text=re.sub("I´m",'I am',text)
    text=re.sub("it’s",'it is',text)
    text=re.sub("she´s",'she is',text)
    text=re.sub("he’s'",'he is',text)
    text=re.sub('I’m','I am',text)
    text=re.sub('I’d','I did',text)
    text=re.sub("he’s'",'he is',text)
    text=re.sub('there’s','there is',text)


    text = re.sub('https?://\S+|www\.\S+', '', text) # removing URL links
    text = re.sub(r"\b\d+\b", "", text) # removing number 
    text = re.sub('<.*?>+', '', text) # removing special characters
    text = text.translate(str.maketrans('','',string.punctuation)) #punctuations
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)

    return text


#Prediction using Multinomial naive bayes
def sentiment_analysis_MNB(input_string):
    clean_input_string = cleaning(input_string)
    lemmatized_string = wordnet_lem.lemmatize(clean_input_string)
    string_array = np.array([lemmatized_string])

    string_vector = MNB_tfidf.transform(string_array)

    predicted = MNB.predict(string_vector)

    if(predicted[0] == 0):
        return "NEGATIVE"
    else:
        return "POSITIVE"
    

#Prediction using n grams Multinomial naive bayes
def sentiment_analysis_MNB2(input_string):
    clean_input_string = cleaning(input_string)
    lemmatized_string = wordnet_lem.lemmatize(clean_input_string)
    string_array = np.array([lemmatized_string])

    string_vector = MNB2_CV2grams.transform(string_array)

    predicted = MNB2.predict(string_vector)

    if(predicted[0] == 0):
        return "NEGATIVE"
    else:
        return "POSITIVE"

def main():
    
    
    # giving a title
    st.title('Sentiment Analysis using naive bayes')
    
    
    # getting the input data from the user
    
    
    input_string = st.text_input('Give Your Opinion to test')
    
    
    # creating a button for Prediction
    
    if st.button('Multinomial NB', key="MNB"):
        prediction = sentiment_analysis_MNB(input_string)
        st.success(prediction)


    if st.button('Multinomial NB 2 Gram', key="MNB2"):
        prediction = sentiment_analysis_MNB2(input_string)
        st.success(prediction)
        
        
    
    
    
    
    
    
if __name__ == '__main__':
    main()




