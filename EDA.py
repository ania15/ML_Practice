import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import plotly.express as px


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df

def show_stats(data):
    st.subheader("Wygląd danych")
    st.write(data.head(5))
    st.subheader("Podstawowe statystyki opisowe")
    st.write(data.describe())

def select_features(data):

    st.sidebar.header('Wybierz cechy:')
    selected_features = st.sidebar.multiselect('Wybierz cechy do wyświetlenia', data.columns[:-1])

    if not selected_features:
        return data

    filtered_data = data.copy()
    for feature in selected_features:
        min_value = st.sidebar.slider(f'Minimalna wartość dla {feature}', data[feature].min(), data[feature].max(), value=data[feature].min())
        max_value = st.sidebar.slider(f'Maksymalna wartość dla {feature}', data[feature].min(), data[feature].max(), value=data[feature].max())
        filtered_data = filtered_data[(filtered_data[feature] >= min_value) & (filtered_data[feature] <= max_value)]
    
    species_list = data['species'].unique()
    selected_species = st.sidebar.multiselect('Wybierz gatunki:', species_list)
    if selected_species:
        filtered_data = filtered_data[filtered_data['species'].isin(selected_species)]
    
    return filtered_data

def show_correlations(df):
    st.subheader('Korelacje między cechami')
    correlation_matrix = df.drop(columns=['species']).corr()
    st.write(correlation_matrix)
    sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f")
    st.pyplot()

def interactive_plot(data):
    st.subheader("Wykres zależności między dwoma wybranymi cechami")
    x_axis = st.selectbox('Wybierz cechę dla osi X:', data.columns[:-2])  
    y_axis = st.selectbox('Wybierz cechę dla osi Y:', data.columns[:-2])
    plot_type = st.radio("Wybierz sposób wyświetlania:", ("Dla każdego gatunku osobno", "Jako całość"), key="plot_type_radio")
    chart_type = st.selectbox('Wybierz typ wykresu:', ['Wykres punktowy', 'Wykres liniowy'])

    if chart_type == 'Wykres punktowy':
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        if plot_type == "Dla każdego gatunku osobno":
            sns.scatterplot(data=data, x=x_axis, y=y_axis, hue='species', palette='Set2')
        else:
            sns.scatterplot(data=data, x=x_axis, y=y_axis)
        plt.title(f'{x_axis} vs {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if plot_type == "Dla każdego gatunku osobno":
            plt.legend(title='Species')
        st.pyplot()
    elif chart_type == 'Wykres liniowy':
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        if plot_type == "Dla każdego gatunku osobno":
            sns.lineplot(data=data, x=x_axis, y=y_axis, hue='species', palette='Set2')
        else:
            sns.lineplot(data=data, x=x_axis, y=y_axis)
        plt.title(f'{x_axis} vs {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if plot_type == "Dla każdego gatunku osobno":
            plt.legend(title='Species')
        st.pyplot()

def interactive_distribution_plot(data):
    st.subheader("Rozkład cech")
    plot_type = st.radio("Wybierz sposób wyświetlania:", ("Dla każdego gatunku osobno", "Jako całość"))
    selected_features = st.multiselect('Wybierz cechy:', data.columns[:-2])  
    
    for feature in selected_features:
        st.subheader(f"Rozkład cechy: {feature}")
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        if plot_type == "Dla każdego gatunku osobno":
            for species in data['species'].unique():
                sns.histplot(data=data[data['species'] == species], x=feature, kde=True, label=species, alpha=0.7)
        else:
            sns.histplot(data=data, x=feature, kde=True, multiple="stack", palette='Set2')
        plt.title(f'Rozkład cechy: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Liczba próbek')
        if plot_type == "Dla każdego gatunku osobno":
            plt.legend(title='Species')
        st.pyplot()

def interactive_boxplot(data):
    st.subheader("Boxploty cech dla każdego gatunku")
    selected_features = st.multiselect('Wybierz cechy:', data.columns[:-2], key='boxplot_multiselect')  

    for feature in selected_features:
        st.subheader(f"Boxplot dla cechy: {feature}")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x='species', y=feature)
        plt.title(f"Boxplot dla cechy: {feature}")
        plt.xlabel('Species')
        plt.ylabel(feature)
        plt.xticks(ticks=[0, 1, 2], labels=['Setosa', 'Versicolor', 'Virginica']) 
        st.pyplot()

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Eksploracyjna analiza danych dla danych IRIS")
    st.write("Zestaw danych Iris zawiera informacje o cechach trzech gatunków irysów: Setosa, Versicolor i Virginica.")
    iris_data = load_data()
    selected_data = select_features(iris_data)
    show_stats(selected_data)
    st.text("")
    interactive_distribution_plot(selected_data)
    st.text("")
    interactive_plot(selected_data)
    st.text("")
    interactive_boxplot(selected_data)
    st.text("")
    show_correlations(selected_data)
    

if __name__ == "__main__":
    main()