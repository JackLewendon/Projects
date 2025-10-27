
import pandas as pd
pd.set_option("display.max_columns", None)

import sys
def exit():
     sys.exit()
     
     
def gap():
     print("\n")
     
import matplotlib.pyplot as plt

import numpy as np

import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')


R_squared = []
R_squared_labels = []

df = pd.read_csv('f1_2024_all_laps.csv')

df = df.dropna(subset=['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed', 'Rainfall']).reset_index(drop=True)
df.drop(columns = ['Time','IsAccurate','DriverNumber',], axis = 1,inplace = True )

events_list = (list(df['Event'].unique()))
team_list = (list(df['Team'].unique()))
compound_list = (list(df['Compound'].unique()))
driver_list = (list(df['Driver'].unique()))

#Encode 1
#Encoder on Event 
encode_event = pd.DataFrame(encoder.fit_transform(df[['Event']]).toarray())

# Put onehotencoder back into original dataframe
df = df.join(encode_event)

#Drop Team Column
df.drop('Event', axis=1, inplace=True)

#Rename these new columns 
df.rename(columns={0:'Abu Dhabi Grand Prix',1:'Australian Grand Prix',2:'Azerbaijan Grand Prix',3:'Bahrain Grand Prix',
                   4:'Belgian Grand Prix',5:'British Grand Prix', 6:'Canadian Grand Prix',7:'Dutch Grand Prix',
                   8:'Emilia Romagna Grand Prix',9:'Hungarian Grand Prix',10:'Italian Grand Prix',11:'Japanese Grand Prix',
                   12:'Las Vegas Grand Prix',13:'Mexico City Grand Prix',14:'Monaco Grand Prix',15:'Saudi Arabian Grand Prix',
                   16:'Singapore Grand Prix',17:'Spanish Grand Prix'},inplace= True) #Alphabetic order and drop the last one

#Drop Last of the column
df = df.drop('Spanish Grand Prix', axis = 1)
print(df)


#Encode 2
#Encoder on Driver 
encode_driver = pd.DataFrame(encoder.fit_transform(df[['Driver']]).toarray())

# Put onehotencoder back into original dataframe
df = df.join(encode_driver)

#Drop Driver Column
df.drop('Driver', axis=1, inplace=True)

#Rename these new columns 
df.rename(columns={0:'ALB', 1:'ALO', 2:'BEA', 3:'BOT', 4:'COL',5:'DOO',6:'GAS',7:'HAM',8:'HUL',9:'LAW',10:'LEC',
                   11:'MAG',12:'NOR',13:'OCO',14:'PER',15:'PIA',16:'RIC',17:'RUS',18:'SAI',
                   19:'SAR',20:'STR',21:'TSU',22:'VER',23:'ZHO'},inplace= True) #Alphabetic order and drop the last one

#Drop Last of the column
df = df.drop('ZHO', axis = 1)

#Encode 3
#Encoder on Team 
encode_team = pd.DataFrame(encoder.fit_transform(df[['Team']]).toarray())

# Put onehotencoder back into original dataframe
df = df.join(encode_team)

#Drop Team Column
df.drop('Team', axis=1, inplace=True)

#Rename these new columns 
df.rename(columns={0:'Alpine',1:'Aston Martin',2:'Ferrari',3:'Haas F1 Team',4:'Kick Sauber',5:'McLaren',
                   6:'Mercedes',7:'RB',8:'Red Bull Racing',9:'Williams'},inplace= True) #Alphabetic order and drop the last one

#Drop Last of the column
df = df.drop('Williams', axis = 1)


#Encode 4
#Encoder on Compound 
encode_compound = pd.DataFrame(encoder.fit_transform(df[['Compound']]).toarray())

# Put onehotencoder back into original dataframe
df = df.join(encode_compound)

#Drop Compound Column
df.drop('Compound', axis=1, inplace=True)

#Rename these new columns 
df.rename(columns={0:'HARD',1:'INTERMEDIATE',2:'MEDIUM',3:'SOFT',4:'WET'},inplace= True) #Alphabetic order and drop the last one

#Drop Last of the column
df = df.drop('WET', axis = 1)

#Encode 5
#Encoder on Rainfall 
encode_rainfall = pd.DataFrame(encoder.fit_transform(df[['Rainfall']]).toarray())

# Put onehotencoder back into original dataframe
df = df.join(encode_rainfall)

#Drop Compound Rainfall
df.drop('Rainfall', axis=1, inplace=True)

#Rename these new columns 
df.rename(columns={0:'No Rain',1:'Rain'},inplace= True) #Alphabetic order and drop the last one

#Drop Last of the column
df = df.drop('Rain', axis = 1)
print(df)

# Regression 

y = df['LapTime_seconds']
x = df.drop(columns=['LapTime_seconds'])
print(list(x))
x = sm.add_constant(x)

#fit linear regression model
results = sm.OLS(y, x).fit()
print(results.summary())

print("results.params: the good stuff")
output = results.params
print(output)
print(type(results.params))
gap()

# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
gap()

R_squared_labels.append("ALL")

###
# Without EVENT and removed the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = x = df.drop(columns=['LapTime_seconds','Bahrain Grand Prix', 'Saudi Arabian Grand Prix',
                         'Australian Grand Prix', 'Japanese Grand Prix', 'Emilia Romagna Grand Prix', 
                         'Monaco Grand Prix', 'Canadian Grand Prix', 'British Grand Prix', 
                         'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 
                         'Italian Grand Prix', 'Azerbaijan Grand Prix', 'Singapore Grand Prix',
                         'Mexico City Grand Prix', 'Las Vegas Grand Prix', 'Abu Dhabi Grand Prix'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Events")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
gap()
print(R_squared)
R_squared_labels.append("Event")


# Without TEAM and removed the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','Alpine', 'Aston Martin', 'Ferrari', 'Haas F1 Team', 
                     'Kick Sauber', 'McLaren', 'Mercedes', 'RB', 'Red Bull Racing'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Team")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("Team")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)


# Without COMPOUND and removed the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','SOFT', 'HARD', 'MEDIUM', 'INTERMEDIATE'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Compound")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("Tire Compound")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)

# Without DRIVER and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','VER', 'LEC', 'RUS', 'PER', 'SAI', 'ALO', 'NOR', 'PIA', 'HAM',
                     'TSU', 'ALB', 'MAG','SAR', 'RIC', 'OCO', 'GAS', 'BOT', 'STR', 'HUL', 'BEA', 'COL',
                     'LAW', 'DOO'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Driver")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("Driver")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)


# Without LapNumber and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','LapNumber'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Lap Number")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("LapNumber")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)


# Without TyreLife and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','TyreLife'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Trye Life")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("TyreLife")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)



# Without AirTemp and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','AirTemp'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without AirTemp")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("AirTemp")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)

# Without TrackTemp and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','TrackTemp'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without TrackTemp")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("TrackTemp")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)


# Without Humidity and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','Humidity'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Humidity")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("Humidity")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)

# Without WindSpeed and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','WindSpeed'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without WindSpeed")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("WindSpeed")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)


# Without Rainfall and remove  the removed column from encoding

y = df['LapTime_seconds']

#define predictor variables
x = df.drop(columns=['LapTime_seconds','No Rain'])
x = sm.add_constant(x)
# exit()

#fit linear regression model
results = sm.OLS(y, x).fit()
gap()
print("Regression Without Rainfall")
print(results.summary())
# exit()  

#print("results.params: the good stuff")
output = results.params
#print(output)


# to get the R squared
print(f"R squared {results.rsquared :.3f}")
R_squared.append(results.rsquared)
R_squared_labels.append("Rainfall")
gap()
R_squared = [round(float(x), 6) for x in R_squared] #Use this after Final regressio  to get clean list of R squared 
print(R_squared)
print(R_squared_labels)


#PLOT 2 
colors = ['steelblue','darkorange','seagreen','firebrick','mediumpurple','gold',
          'teal','slategray','crimson','olive','dodgerblue','indianred','orchid',
          'sienna','royalblue','darkcyan','peru','deeppink','limegreen','chocolate',
          'cadetblue','navy','darkmagenta','coral','darkkhaki'
]
plt.figure(figsize=(10, 10))
plt.bar(R_squared_labels, R_squared, color=colors, edgecolor='black')
plt.title("R² (drop-one vs all 11) model impact comparison -  Spanish Grand Prix'",
fontsize=16, fontweight='bold')
plt.xlabel("Dropped Feature", fontsize=20)
plt.ylabel("R² (model performance)", fontsize=20)
plt.xticks(range(len(R_squared_labels)), R_squared_labels, fontsize=12, rotation=90, fontweight='bold')
plt.yticks(fontsize =12,fontweight='bold')
plt.ylim(0.2, 1.01)  # set y range
for i in range(len(R_squared_labels)):
    plt.text(i, R_squared[i] + 0.0025, f"{R_squared[i]:.5f}", 
             ha='center', fontsize=10,fontweight='bold')

plt.tight_layout()
plt.show()


print(R_squared)


