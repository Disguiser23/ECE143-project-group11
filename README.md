# ECE143-project-group11
## Team members: Linus Grasel, Young Truong, Girish Gowtham Aravindan, Sai Ding 
## Problem:
- Identification of leading factors in EV adoption across the United States
## Datasets:
- We will be combining different sources of Electric Vehicle Sales Data by States from 2015-2018 (such
as https://autoalliance.org/energy-environment/advanced-technology-vehicle-sales-dashboard/). Baseline
data will be associated with other datasets, such as data on EV Charging Stations
(https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/all/) and Vehicle features data
(https://afdc.energy.gov/data/) to investigate possible correlations.
## Proposed Solution and Real world Application :
- We propose to perform statistical analysis and data correlation to identify the most important features
that drive EV adoption in the United States, whether directly or indirectly. We will look at various
factors, such as number of charging stations, gas and electricity price, road quality, gas and
electricity price, average education level, state purchase incentives, ... to determine which ones are
most highly correlated with EV adoption. Our goal is to contribute to the shift towards a zero-emission
environment.
- We might attempt to implement different machine learning models to see if we can predict which states
are more likely to see EV adoption over the coming years and can be leveraged by local representatives
to build EV infrastructure. 

## File Structure

```
Root
|
+----ECE 143.csv                   #Data we used in our model
|
|    getData.py                   #Code to preprocess data we used
|    linearRegression.py          #Code to do linear regression analysis
|    decisionTree.py              #Code to do prediction based on decsion tree model
|    lassoPrediction.py           #Code to do prediction based on lasso model
|
+----main.ipynb                   # The main jupyterNotebook to show the results and graphics
|
+----shape files                  # map overlays to plot sales data on US map
+----decisionTree                 # Image of decision trees 
+----plots                        # US sales map and correlation plots
```
## How to run the code
- Python version: 3.7.3
- Run the getData.py to get the processed_data from raw_data file.
- Run the linearRegression.py to get the results and graphs from linearRegression analysis.
- Run the decisionTree.py to do prediction based on decsion tree model
- Run the lassoPrediction.py to do prediction based on lasso model
- Run the main.ipynb to get all the graphics in this project
## Third-party modules
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. sklearn
6. geopandas
7. plotly
