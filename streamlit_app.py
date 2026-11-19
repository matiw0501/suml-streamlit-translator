import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os
# zaczynamy od zaimportowania bibliotek

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')
# streamlit jest wykorzystywany do tworzenia aplikacji
# z tego powodu dobrą praktyką jest informowanie użytkownika o postępie, błędach, etc.

# Inne przykłady do wypróbowania:
# st.balloons() # animowane balony ;)
# st.error('Błąd!') # wyświetla informację o błędzie
# st.warning('Ostrzeżenie, działa, ale chyba tak sobie...')
# st.info('Informacja...')
# st.success('Udało się!')

# st.spinner()
# with st.spinner(text='Pracuję...'):
    # time.sleep(2)
    # st.success('Done')
# możemy dzięki temu "ukryć" późniejsze ładowanie aplikacji
col1, col2 = st.columns([1,5], vertical_alignment="center")
with col1:
    st.image("translateLogo.png", width=100)
with col2:
    st.title('Lab05. Streamlit - my first translate app')
# title, jak sama nazwa wskazuje, używamy do wyświetlenia tytułu naszej aplikacji

st.header('Wprowadzenie do zajęć')
# header to jeden z podtytułów wykorzystywnaych w Streamlit

st.subheader('O Streamlit')
# subheader to jeden z podtytułów wykorzystywnaych w Streamlit

st.text('To przykładowa aplikacja z wykorzystaniem Streamlit')
# text używamy do wyświetlenia dowolnego tekstu. Można korzystać z polskich znaków.

st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')
# write używamy również do wyświetlenia tekstu, różnica polega na formatowaniu.

st.code("st.write()", language='python')
# code może nam się czasami przydać, jeżeli chcielibyśmy pokazać np. klientowi fragment kodu, który wykorzystujemy w aplikacji

with st.echo():
    st.write("Echo")
# możemy też to zrobić prościej używając echo - pokazujemy kod i równocześnie go wykonujemy

df = pd.read_csv("DSP_4.csv", sep = ';')
st.dataframe(df)
# musimy tylko pamiętać o właściwym określeniu separatora (w tym wypadku to średnik)
# masz problem z otworzeniem pliku? sprawdź w jakim katalogu pracujesz i dodaj tam plik (albo co bardziej korzystne - zmień katalog pracy)
# os.getcwd() # pokaż bieżący katalog
# os.chdir("") # zmiana katalogu

st.header('Przetwarzanie języka naturalnego')
st.write("Aplikacja służy do tłumaczenia tekstów z języka angielskiego na niemiecki. Wybierz odpowiednią opcje, wprowadź w polu tekstowym a następnie wciśnij " \
"'ctr + enter' a zobaczysz tłumaczenie")

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator

@st.cache_resource
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    return tokenizer, model

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumacz tekstu transformers (eng - de)",
        "Tlumacz tekstu deep-translator (eng - de)"
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = load_sentiment()
        answer = classifier(text)
        st.write(answer)
        st.success('Gratulacje! Z powodzeniem uruchomiłeś wydźwiek emocjonalny tekstu')
if option == "Tłumacz tekstu transformers (eng - de)":
    text = st.text_area(label="Write text in english to translate for german (transformers)")
    if text:
        try:
            with st.spinner("Translating... "): #sprawdzanie czy dziala
                time.sleep(2)
                tokenizer, model = load_translator()
                inputs = tokenizer(text, return_tensors = "pt")
                outputs = model.generate(**inputs)
            st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
            st.success('Gratulacje! Z powodzeniem uruchomiłeś tłumaczenie')
        except Exception as e:
            st.error(f"Something wrong with translating: {e}")
    
if option == "Tlumacz tekstu deep-translator (eng - de)":
    text = st.text_area(label="Write text in english to translate for german (deep-transalte)")
    if text:
        try:
            st.write(GoogleTranslator(source="en", target='de').translate(text))
            st.success('Gratulacje! Z powodzeniem uruchomiłeś tłumaczenie')
        except Exception as e:
            st.error(f"Błąd w tłumaczeniu: {e}")



st.header("s31224")

st.subheader('Zadanie do wykonania')
st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
st.write('🐞 Na końcu umieść swój numer indeksu')
st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')





