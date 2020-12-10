import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import geopandas as gpd
import plotly.graph_objs as graphObj
from plotly.offline import plot, iplot
import sys
import adjustText as aT
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import statistics as stats

df = pd.read_csv('ECE 143.csv')
df.set_index(["State"], inplace=True)
if 'average_sales' not in df:
    df.insert(5, 'average_sales', df.iloc[:,1:5].mean(axis=1).to_frame())
    df['average_sales'] = df['average_sales'].astype(int)
# Adjust columns names
df.rename(columns={"Avg/C": "Average temperature(Celsius)",
                   "Median Household Income\t $": "Median Household Income",
                   "% Libertarian/ Independent Representation": "% Libertarian/Independent Representation",
                   "Avg gasoline price per gallon": "Average gasoline price per gallon",
                   "COMMUTE TIME": "Commute time",
                   "PUBLIC TRANSIT USAGE": "Public transit usage",
                   "ROAD QUALITY": "Road quality",
                   "BRIDGE QUALITY": "Bridge quality"
                   }, inplace=True)

# Correlation plot between average sales and continuous features
avg_sales = df['average_sales']
# avg_sales = df['Avg % of Total Sales 2015-2018']
# avg_sales.apply(lambda x : (x-avg_sales.mean())/avg_sales.std())
cont_features = ['Average temperature(Celsius)', 'Commute time','Public transit usage','Road quality','Bridge quality','Average retail price (cents/kWh)','Average gasoline price per gallon',
             '% High school graduate\nor higher',"% Bachelor's degree\nor higher",'Advanced degree','Democratic Representation','Republican Representation',
             '% Green Representation','% Libertarian/Independent Representation','Median Household Income','Charging Locations',
             'Charging Outlets','Outlets Per Location','EVs to Charging Outlets']
cont_cols = df.loc[:,cont_features]
cont_cols.columns = cont_features
fig, axs = plt.subplots(9, 2,figsize=(15,20),constrained_layout=True)
fig.suptitle("Correlation with Average EV Sales",fontsize=22)
# fig.suptitle("Correlation with Average EV market share",fontsize=22)
for i in range(18):
    r = i//2
    c = i%2
    # get feature, normalize and plot
    col = cont_cols.iloc[:,i-1].astype(float)
    col=col.apply(lambda x : (x-col.mean())/col.std())
    axs[r, c].plot(col.iloc[:], avg_sales.iloc[:], 'bo')
    # set plot title, showing correlation value
    corr = avg_sales.corr(col)
    axs[r, c].set_title("{}. r: {:2f}".format(col.name,corr))
    # format plot
    axs[r, c].xaxis.set_visible(False)
    axs[r, c].yaxis.set_visible(False)
    x0,x1 = axs[r, c].get_xlim()
    y0,y1 = axs[r, c].get_ylim()
    axs[r, c].set_aspect(abs(x1-x0)/abs(y1-y0))
plt.show()
Y = df['Avg % of Total Sales 2015-2018']
Y_2018 = df["EV Sales\n2018 % of Total"]
# Y = df["average_sales"]/df["Population"]
X = df.loc[:,['Commute time','Public transit usage','Charging Locations', 'Charging Outlets', 'Democratic Representation']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# scale data
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(x_train)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Fit data and print coefficients
lrModel = LinearRegression()
x = X_scaled
y = y_train
lrModel.fit(x,y)
print("Linear Regression score: " + str(lrModel.score(x,y)))
print("Weights:")
for i in range(len(X.columns)):
    print("{}: {}".format(X.columns[i], lrModel.coef_[i]))
print("Bias: " + str(lrModel.intercept_))

x_test_scaled = min_max_scaler.fit_transform(x_test)
pred = lrModel.predict(x_test_scaled)
result = pd.DataFrame({ 'true_labels': y_test, 'prediction': pred })
print("\nTest set performance")
print(result)
print("Test score: " + str(lrModel.score(x_test_scaled,y_test)))
print("Mean squared error: {}".format(stats.mean([(y_test[i]-pred[i])**2 for i in range(len(pred))])))


print("\nPredict 2018 Sales")
X = df.loc[:,['Commute time','Public transit usage','Charging Locations', 'Charging Outlets', 'Democratic Representation']]
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
pred = lrModel.predict(X_scaled)
# result = pd.DataFrame({ 'true_labels': Y_2018, 'prediction': pred })
# print(result)
print("Prediction score: " + str(lrModel.score(X_scaled,Y_2018)))
print("Mean squared error: {}".format(stats.mean([(Y_2018[i]-pred[i])**2 for i in range(len(pred))])))


sns.set(color_codes=True)
plt.figure(figsize=(15,10))
plt.plot(Y_2018, 'bx', label='truth',markersize=10.,marker='o',c=sns.xkcd_rgb['orange red'])
plt.plot(pred, 'rx', label='prediction',markersize=10.,marker='o',c=sns.xkcd_rgb['dodger blue'])
plt.legend(labels=['True','Prediction'],loc='best',prop = {'size':15})

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
plt.suptitle("2018 prediction performance",fontsize=22)
plt.xticks(rotation=90,fontsize = 15,alpha=2.0)
plt.yticks(fontsize = 15,alpha=2.0)
plt.xlabel('State',font2)
plt.ylabel('Value',font2)
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from IPython.display import Image, display
tree_reg_2 = DecisionTreeRegressor(max_depth=2)
tree_reg_2.fit(x_train,y_train)
export_graphviz( tree_reg_2,
out_file="ev_tree_max_depth_2.dot", feature_names=list(x.columns))

display(Image(filename='ev_tree_2_avg_sales.png'))
display(Image(filename='ev_tree_2_market_share.png'))
print("\nTest set performance")
pred = tree_reg_2.predict(x_test)
result = pd.DataFrame({ 'true_labels': y_test, 'prediction': pred })
print(result)

print("Mean squared error: {}".format(stats.mean([(y_test[i]-pred[i])**2 for i in range(len(pred))])))