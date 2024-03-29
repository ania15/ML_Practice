import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
 
# Wczytanie danych Iris
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df
 
# Funkcja do wyświetlania podstawowych statystyk opisowych
def show_basic_stats(data):
    st.subheader("Podstawowe statystyki opisowe")
    st.write(data.describe())
 
# Funkcja do wizualizacji rozkładu zmiennych
def show_distribution_plots(data):
   st.subheader("Rozkład zmiennych")
   for column in data.columns[:-1]:
       plt.figure(figsize=(8, 6))
       sns.histplot(data=data, x=column, hue='target', kde=True)
       plt.title(f"Dystrybucja zmiennej: {column}")
       plt.legend(labels=['Setosa','Versicolor', 'Virginica'])
       st.pyplot(plt)  # Przekazanie aktualnego wykresu jako argument
 
# Funkcja główna aplikacji
def main():
    st.title("Aplikacja do Eksploracyjnej Analizy Danych (EDA) dla danych Iris")

    # Wczytanie danych
    iris_data = load_iris_data()

    # Wyświetlenie podstawowych statystyk opisowych
    show_basic_stats(iris_data)
 
    # Wyświetlenie wizualizacji rozkładu zmiennych
    show_distribution_plots(iris_data)
 
# Uruchomienie aplikacji
if __name__ == "__main__":
   main()