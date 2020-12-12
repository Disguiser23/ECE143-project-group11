import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import geopandas as gpd
# import plotly.graph_objs as graphObj
# from plotly.offline import plot, iplot
import sys
import adjustText as aT
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
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
        df.insert(5, 'average_sales', df.iloc[:, 1:5].mean(axis=1).to_frame())
        df['average_sales'] = df['average_sales'].astype(int)
    return df
def get_sales_plot(df):
    '''
    # Plot state sales data
    :param df: preprocessed data
    :return: pd.Dataframe
    '''
    sales_data = df.iloc[:, 1:6]
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

def get_us_map(df):
    assert isinstance(df, pd.DataFrame)
    stateDict = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY',
    }


    df['StateAbbreviation'] = df['State'].map(stateDict)

    if 'average_sales' not in df:
        df.insert(5, 'average_sales', df.iloc[:, 2:6].mean(axis=1).to_frame())
        df.insert(6, 'total_sales', df.iloc[:, 2:6].sum(axis=1).to_frame())
        df['average_sales'] = df['average_sales'].astype(int)
        df['total_sales'] = df['total_sales'].astype(int)
        df['averageSales_Population'] = df['average_sales'] / df['Population']
        df['averageSales_TotalSales'] = df['average_sales'] / df['total_sales']

    dfSub = df[['StateAbbreviation', 'EV Sales\n2018']]

    us_states = gpd.read_file("shape files/cb_2018_us_state_20m.shp")
    us_states.drop(index=[7, 25, 48], inplace=True)
    df.set_index(["State"], inplace=True)
    us_states_plot = pd.merge(us_states, dfSub, left_on='STUSPS', right_on='StateAbbreviation')
    us_states_plot["center"] = us_states_plot["geometry"].centroid
    us_states_plot_points = us_states_plot.copy()
    us_states_plot_points.set_geometry("center", inplace=True)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["whitesmoke", "lightgrey", "mistyrose", "wheat", "coral", "beige", "peru", "crimson"])
    bounds = [0, 250, 500, 750, 1000, 5000, 10000, 15000, 150000]
    norm = BoundaryNorm(bounds, cmap.N)
    texts = []
    ax = us_states_plot.geometry.plot(figsize=(20, 16), color="whitesmoke", edgecolor="lightgrey", linewidth=0.5)
    for x, y, label in zip(us_states_plot_points.geometry.x, us_states_plot_points.geometry.y,
                           us_states_plot_points["STUSPS"]):
        texts.append(plt.text(x, y, label, color='grey', fontsize=12))
    aT.adjust_text(texts, force_points=0.3, force_text=1, expand_points=(1, 1), expand_text=(1, 1),
                   arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))
    us_states_plot.plot(column='EV Sales\n2018', legend_kwds={'shrink': 0.5}, legend=True, ax=ax, cmap=cmap, norm=norm)
    ax.axis('off')
    ax.set_title("EV sales actual in 2018")
    None
    sales_2018 = np.array(df[['EV Sales\n2018']])
    sales_2015 = np.array(df[['EV Sales\n2015']])
    df['percent_Increase_in_sales_2015_2018'] = (sales_2018 - sales_2015) / (sales_2018)
    dfPrct = df[["StateAbbreviation", "percent_Increase_in_sales_2015_2018"]]
    us_states_plot = pd.merge(us_states_plot, dfPrct, on='StateAbbreviation')
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["whitesmoke", "lightgrey", "mistyrose", "wheat", "coral", "beige", "crimson"])
    bounds = [-0.5, 0, 0.1, 0.25, 0.4, 0.65, 0.8, 1]
    norm = BoundaryNorm(bounds, cmap.N)
    texts = []
    ax = us_states_plot.geometry.plot(figsize=(20, 16), color="whitesmoke", edgecolor="lightgrey", linewidth=0.5)
    for x, y, label in zip(us_states_plot_points.geometry.x, us_states_plot_points.geometry.y,
                           us_states_plot_points["STUSPS"]):
        texts.append(plt.text(x, y, label, color='grey', fontsize=12))
    aT.adjust_text(texts, color='green', force_points=0.3, force_text=1, expand_points=(1, 1), expand_text=(1, 1),
                   arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))
    us_states_plot.plot(column='percent_Increase_in_sales_2015_2018', legend_kwds={'shrink': 0.5}, legend=True, ax=ax,
                        cmap=cmap, norm=norm)
    ax.axis('off')
    ax.set_title("EV increase in sales from 2015 to 2018 (%)")
    plt.show()
    None
if __name__ == '__main__':
    df = pd.read_csv('ECE 143.csv')
    get_us_map(df)
    dx = pd.read_csv('ECE 143.csv')
    dx.set_index(["State"], inplace=True)
    dx = get_average_sales(dx)
    get_sales_plot(dx)



