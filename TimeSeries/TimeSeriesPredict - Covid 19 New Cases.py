import pandas as pd
from numpy import around
from statsmodels.tsa.statespace.sarimax import SARIMAX


# definindo arquivo de dados
coviddata = pd.read_csv('owid-covid-data.csv')

# extraindo apenas os dados do mundo total
data = pd.DataFrame(coviddata[coviddata['iso_code'].str.startswith('OWID_WRL')])
data['date'] = pd.to_datetime(data['date'])
date_cases = data[['date', 'new_cases']]
date_cases.set_index('date', inplace=True)

t_train = date_cases[date_cases.index < '2021-02-01'].index.values
t_test = date_cases[date_cases.index >= '2021-02-01'].index.values
X_train = date_cases[date_cases.index < '2021-02-01'].values
X_test = date_cases[date_cases.index >= '2021-02-01'].values

date_cases.index = pd.DatetimeIndex(date_cases.index).to_period('D')
Model = SARIMAX(X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
results = Model.fit(disp=0)


def predict():
    dias = input('Digite o número de dias a ser previsto: ')
    for day in range(1, int(dias) + 1):
        print(day, '->', around(results.predict(start=len(date_cases) - 1 + day, end=len(date_cases) - 1 + day)
                                         ))


predict()
