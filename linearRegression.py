import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import statistics as stats

def preProcessing(df):
    '''

    :param df: Original data
    :return: df to be used in LinearRegression
    '''
    assert isinstance(df,pd.DataFrame)
    df.set_index(["State"], inplace=True)
    # drop outlier
    df = df.drop("California")
    df = df.drop("District of Columbia")
    df.rename(columns={"Avg/C": "Average temperature(Celsius)",
                       "Median Household Income\t $": "Median Household Income",
                       "% Libertarian/ Independent Representation": "% Libertarian/Independent Representation",
                       "Avg gasoline price per gallon": "Average gasoline price per gallon",
                       "COMMUTE TIME": "Commute time",
                       "PUBLIC TRANSIT USAGE": "Public transit usage",
                       "ROAD QUALITY": "Road quality",
                       "BRIDGE QUALITY": "Bridge quality"
                       }, inplace=True)
    return df
def correlationAnalysis(df):
    '''
    # Correlation plot between average sales and continuous features
    :param df: preprocessed data
    '''
    assert isinstance(df, pd.DataFrame)
    avg_sales = df['Avg % of Total Sales 2015-2018']
    cont_features = ['Average temperature(Celsius)', 'Commute time', 'Public transit usage', 'Road quality',
                     'Bridge quality', 'Average retail price (cents/kWh)', 'Average gasoline price per gallon',
                     '% High school graduate\nor higher', "% Bachelor's degree\nor higher", 'Advanced degree',
                     'Democratic Representation', 'Republican Representation',
                     '% Green Representation', '% Libertarian/Independent Representation', 'Median Household Income',
                     'Charging Locations',
                     'Charging Outlets', 'Outlets Per Location', 'EVs to Charging Outlets']
    cont_cols = df.loc[:, cont_features]
    cont_cols.columns = cont_features
    fig, axs = plt.subplots(9, 2, figsize=(15, 20), constrained_layout=False)
    fig.suptitle("Correlation with Average EV market share", fontsize=22)
    for i in range(18):
        r = i // 2
        c = i % 2
        # get feature, normalize and plot
        col = cont_cols.iloc[:, i - 1].astype(float)
        col = col.apply(lambda x: (x - col.mean()) / col.std())
        axs[r, c].plot(col.iloc[:], avg_sales.iloc[:], 'bo')
        # set plot title, showing correlation value
        corr = avg_sales.corr(col)
        axs[r, c].set_title("{}. r: {:2f}".format(col.name, corr))
        # format plot
        axs[r, c].xaxis.set_visible(False)
        axs[r, c].yaxis.set_visible(False)
        x0, x1 = axs[r, c].get_xlim()
        y0, y1 = axs[r, c].get_ylim()
        axs[r, c].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    plt.show()
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
def testPerformance(df):
    '''
    # Test the correlation performance for each features
    :param df: preprocessed data
    '''
    assert isinstance(df, pd.DataFrame)
    df = get_average_sales(df)
    # To try using ev_market_share or avg_sale/population as labels, uncomment the line below
    Y = df["average_sales"]
    X = df.loc[:,
        ['Commute time', 'Public transit usage', 'Charging Locations', 'Charging Outlets', 'Democratic Representation']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(x_train)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Fit data and print coefficients
    lrModel = LinearRegression()
    x = X_scaled
    y = y_train
    lrModel.fit(x, y)
    print("Linear Regression score: " + str(lrModel.score(x, y)))
    print("Weights:")
    for i in range(len(X.columns)):
        print("{}: {}".format(X.columns[i], lrModel.coef_[i]))
    print("Bias: " + str(lrModel.intercept_))

    x_test_scaled = min_max_scaler.fit_transform(x_test)
    pred = lrModel.predict(x_test_scaled)
    result = pd.DataFrame({'true_labels': y_test, 'prediction': pred})
    print("\nTest set performance")
    print(result)
    plt.plot(pred, 'rx', label='prediction')
    plt.plot(y_test, 'bx', label='truth')
    plt.suptitle("Test performance", fontsize=22)
    plt.show()
    print("Mean squared error: {}".format(stats.mean([(y_test[i] - pred[i]) ** 2 for i in range(len(pred))])))

if __name__ == '__main__':
    df = pd.read_csv('ECE 143.csv')
    df = preProcessing(df)
    correlationAnalysis(df)
    testPerformance(df)

