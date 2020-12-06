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
+----raw_data
|
+----processed_data
|
+----scripts
|    getData.py

```
## How to run the code

## Third-party modules
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. sklearn
