import pandas as pd

class DataLoader:
    '''
    Класс для загрузки данных из CSV-файла.
    '''
    
    def __init__(self, url):
        '''
        Инициализирует класс DataLoader.

        Args:
            url (str): URL CSV-файла.
        '''
        self.url = url


    def load_data(self) -> pd.DataFrame:
        '''
        Загружает данные из CSV-файла.

        Returns:
            pd.DataFrame: Загруженный DataFrame.
        '''
        try:
            df = pd.read_csv(self.url)
            return df
        except Exception as e:
            print(f'Ошибка при загрузке данных: {e}')
            return None


if __name__ == '__main__':
    url = 'https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv?raw=true'
    loader = DataLoader(url)
    titanic_df = loader.load_data()
    
    if titanic_df is not None:
        print(titanic_df.head())