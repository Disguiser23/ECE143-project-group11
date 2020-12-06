import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''
Code to get data be used
'''
def get_average_sales(df):
    '''
    Compute average sales and add to df
    :param df: Original data
    :return: pd.Dataframe
    '''
    if 'average_sales' not in df:
        df.insert(4, 'average_sales', df.iloc[:, 0:4].mean(axis=1).to_frame())
        df['average_sales'] = df['average_sales'].astype(int)
    return df
def get_sales_plot(df):
    '''
    # Plot state sales data
    :param df: preprocessed data
    :return: pd.Dataframe
    '''
    sales_data = df.iloc[:, 0:5]
    sales_data['State'] = sales_data.index.get_level_values('State')
    x = pd.melt(sales_data, id_vars="State", var_name="years", value_name="sale")
    y = x.groupby(["State", "years"]).sum()
    y['State'] = y.index.get_level_values('State')
    y['years'] = y.index.get_level_values('years')
    plt.figure(figsize=(15, 20))
    sns.barplot(x='sale', y='State', hue="years", data=y)
    plt.title('EV sales of US')
    plt.savefig('EV_sales.png', transparent=True)
    plt.show()

df = pd.read_csv('ECE 143.csv')
df.set_index(["State"], inplace=True)
# drop outlier
df = df.drop("California")
df = df.drop("District of Columbia") # advanced degree outlier

df=get_average_sales(df)
get_sales_plot(df)

