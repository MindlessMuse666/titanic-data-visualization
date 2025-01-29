import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from matplotlib.font_manager import FontProperties
from data_loader import DataLoader


class PlotGenerator:
    '''
    Класс для генерации различных графиков на основе DataFrame.
    '''
    
    def __init__(self, df: pd.DataFrame):
        '''
        Инициализирует PlotGenerator с DataFrame.

        Args:
            df (pd.DataFrame): DataFrame для генерации графиков.
        '''
        self.df = df
        self.font = FontProperties(fname='static/times.ttf')
        
        plt.rcParams['font.family'] = 'Times New Roman' # Загрузка шрифта


    def create_bar_chart(self, x_col: str, y_col: str = None, title: str = None, xlabel: str = None, ylabel: str = None, color: str = None, xticks_labels: dict = None):
        '''
        Создает столбчатую диаграмму.

        Args:
            x_col (str): Имя столбца для оси x.
            y_col (str, optional): Имя столбца для оси y. Defaults to None.
            title (str, optional): Заголовок графика. Defaults to None.
            xlabel (str, optional): Подпись для оси x. Defaults to None.
            ylabel (str, optional): Подпись для оси y. Defaults to None.
            color (str, optional): Цвет столбцов. Defaults to None.
            xticks_labels (dict, optional): Словарь для замены подписей оси x. Defaults to None.
        '''
        plt.figure(figsize=(8, 6), num=title)
        
        if y_col:
            ax = sns.barplot(x=x_col, y=y_col, data=self.df, color=color)
        else:
            ax = sns.countplot(x=x_col, data=self.df, color=color)
             
        if title:
            plt.title(title, fontproperties=self.font, pad=20)
        if xlabel:
            plt.xlabel(xlabel, fontproperties=self.font, labelpad=10)
        if ylabel:
            plt.ylabel(ylabel, fontproperties=self.font, labelpad=10)
        if xticks_labels:
            ax.set_xticks(ax.get_xticks())
            labels = [xticks_labels.get(item.get_text(), item.get_text()) for item in ax.get_xticklabels()]
            ax.set_xticklabels(labels, fontproperties=self.font)

        plt.show()


    def create_histogram(self, column: str, title: str = None, xlabel: str = None, ylabel: str = None, color: str = None):
        '''
        Создает гистограмму.

        Args:
            column (str): Имя столбца для гистограммы.
            title (str, optional): Заголовок графика.
            xlabel (str, optional): Подпись для оси x.
            ylabel (str, optional): Подпись для оси y.
            color (str, optional): Цвет гистограммы.
        '''
        plt.figure(figsize=(8, 6), num=title)
        sns.histplot(self.df[column].dropna(), kde=True, color=color)
        
        if title:
            plt.title(title, fontproperties=self.font, pad=20)
        if xlabel:
            plt.xlabel(xlabel, fontproperties=self.font, labelpad=10)
        if ylabel:
            plt.ylabel(ylabel, fontproperties=self.font, labelpad=10)
            
        plt.show()
        
        # Анализ гистограммы
        mean_age = self.df[column].mean()
        median_age = self.df[column].median()
        
        print(f'\nСредний возраст пассажиров: {mean_age:.2f}')
        print(f'Медианный возраст пассажиров: {median_age:.2f}')
        print('Вывод: распределение возрастов на борту Титаника показывает, что большинство пассажиров были в возрасте от 20 до 40 лет, при этом медианный возраст около 28 лет.\n')


    def create_pie_chart(self, column: str, title: str = None):
        '''
        Создает круговую диаграмму.

        Args:
            column (str): Имя столбца для круговой диаграммы.
            title (str, optional): Заголовок графика. Defaults to None.
        '''
        plt.figure(figsize=(8, 6), num=title)
        counts = self.df[column].value_counts()
        labels = ['мужчины' if label == 'male' else 'женщины' for label in counts.index]
        colors = ['skyblue' if label == 'мужчины' else 'lightcoral' for label in labels]
        wedges, texts, autotexts = plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
                                         wedgeprops={'edgecolor': 'black', 'linewidth': 1}) # Добавление обводки
        
        if title:
            plt.title(title, fontproperties=self.font, pad=20)
            
        plt.show()


    def create_boxplot(self, x_col: str, y_col: str, title: str = None, xlabel: str = None, ylabel: str = None):
        '''
        Создает ящик с усами.
        
        Args:
            x_col (str): Имя столбца для оси x.
            y_col (str): Имя столбца для оси y.
            title (str, optional): Заголовок графика.
            xlabel (str, optional): Подпись для оси x.
            ylabel (str, optional): Подпись для оси y.
        '''
        plt.figure(figsize=(8, 6), num=title)
        sns.boxplot(x=x_col, y=y_col, data=self.df, hue=x_col, legend=False)
        
        if title:
            plt.title(title, fontproperties=self.font, pad=20)
        if xlabel:
            plt.xlabel(xlabel, fontproperties=self.font, labelpad=10)
        if ylabel:
            plt.ylabel(ylabel, fontproperties=self.font, labelpad=10)
            
        plt.show()


    def create_heatmap(self, title: str = None):
        '''
        Создает тепловую карту корреляции.

        Args:
           title (str, optional): Заголовок графика.
        '''
        plt.figure(figsize=(10, 8), num=title)
        numeric_df = self.df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        
        if title:
            plt.title(title, fontproperties=self.font, pad=20)
            
        plt.show()


    def create_scatter_plot(self, x_col: str, y_col: str, title: str = None, xlabel: str = None, ylabel: str = None):
        '''
        Создает диаграмму рассеяния.

        Args:
           x_col (str): Имя столбца для оси x.
           y_col (str): Имя столбца для оси y.
           title (str, optional): Заголовок графика.
           xlabel (str, optional): Подпись для оси x.
           ylabel (str, optional): Подпись для оси y.
        '''
        plt.figure(figsize=(8, 6), num=title)
        sns.scatterplot(x=x_col, y=y_col, data=self.df, color='orange', alpha=0.6)
        
        if title:
            plt.title(title, fontproperties=self.font, pad=20)
        if xlabel:
            plt.xlabel(xlabel, fontproperties=self.font, labelpad=10)
        if ylabel:
            plt.ylabel(ylabel, fontproperties=self.font, labelpad=10)
            
        plt.show()


    def create_interactive_scatter(self, x_col: str, y_col: str, color_col: str, size_col: str, title: str = None):
        '''
        Создает интерактивную диаграмму рассеяния с помощью Plotly.

        Args:
            x_col (str): Имя столбца для оси x.
            y_col (str): Имя столбца для оси y.
            color_col (str): Имя столбца для цвета точек.
            size_col (str): Имя столбца для размера точек.
            title (str, optional): Заголовок графика.
        '''
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, size=size_col,
                        hover_data=self.df.columns, title=title)
        
        fig.show()


    def create_subplots(self, x_col_1: str, x_col_2: str, y_col: str, title: str = None):
        '''
        Создает подграфики.

        Args:
            x_col_1 (str): Имя первого столбца для оси x.
            x_col_2 (str): Имя второго столбца для оси x.
            y_col (str): Имя столбца для оси y.
            title (str, optional): Заголовок графика.
        '''
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), num=title)
        sns.barplot(x=x_col_1, y=y_col, data=self.df, ax=axes[0])
        sns.barplot(x=x_col_2, y=y_col, data=self.df, ax=axes[1])
        
        if title:
            fig.suptitle(title, fontproperties=self.font)
            
        plt.tight_layout()
        plt.show()
        
        
    def analyze_survival_by_class(self, class_col: str, survived_col: str, title: str = None):
        '''
        Анализирует и выводит выживаемость по классам.

        Args:
            class_col (str): Имя столбца, содержащего информацию о классе.
            survived_col (str): Имя столбца, содержащего информацию о выживании.
            title (str, optional): Заголовок графика.
        '''
        survival_rate = self.df.groupby(class_col)[survived_col].mean()
        
        print('Выживаемость по классам:')
        for pclass, rate in survival_rate.items():
            print(f'Класс {pclass}: {rate:.2%}')
            
        highest_survival_class = survival_rate.idxmax()
        highest_survival_rate = survival_rate.max()
        print(f'\nСамая высокая выживаемость ({highest_survival_rate:.2%}) была в классе {highest_survival_class}.')
        
        self.create_bar_chart(x_col=class_col, y_col=survived_col, title=title, xlabel='Класс', ylabel='Выживаемость', color='skyblue')


if __name__ == '__main__':
    url = 'https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv?raw=true'
    loader = DataLoader(url)
    titanic_df = loader.load_data()
    
    if titanic_df is not None:
        plotter = PlotGenerator(titanic_df)
        plotter.create_bar_chart(x_col='survived', 
                                 title='Выжившие и погибшие пассажиры', 
                                 xlabel='Статус', 
                                 ylabel='Количество', 
                                 color='skyblue', 
                                 xticks_labels={0:'Погибшие', 1:'Выжившие'})
        plotter.create_histogram(column='age', 
                                title='Распределение возрастов', 
                                xlabel='Возраст', 
                                ylabel='Количество', 
                                color='skyblue')
        plotter.analyze_survival_by_class(class_col='pclass',
                                         survived_col='survived',
                                         title='Выживаемость по классам')
        plotter.create_pie_chart(column='sex', title='Половое распределение')
        plotter.create_boxplot(x_col='pclass', y_col='fare', title='Распределение стоимости билетов по классам', xlabel='Класс', ylabel='Стоимость')
        plotter.create_heatmap(title='Корреляция между числовыми признаками')
        plotter.create_scatter_plot(x_col='age', y_col='fare', title='Возраст и стоимость билета', xlabel='Возраст', ylabel='Стоимость')
        plotter.create_interactive_scatter(x_col='age', y_col='fare', color_col='pclass', size_col='survived', title='Интерактивная карта выживаемости по классам')
        plotter.create_subplots(x_col_1='sibsp', x_col_2='parch', y_col='survived', title='Влияние наличия семьи на выживаемость')
        plotter.create_bar_chart(x_col='embark_town', y_col='survived', title='Выживаемость по портам посадки', xlabel='Порт посадки', ylabel='Выжившие', color='skyblue')
