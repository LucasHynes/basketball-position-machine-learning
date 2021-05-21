import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from matplotlib import pyplot as plt
min_max_scaler = preprocessing.MinMaxScaler()

#size of resulting plot figure
fig_size = 12

#importing the data from the graph
basketball_data = pd.read_csv("nba-data-historical.csv", engine = 'python')

#cleans up percent values to double values removing '%' symbol
for column_name, column_data in basketball_data.iteritems():
    if '%' in column_name:
        column_data = [i*0.01 for i in column_data]
        basketball_data[column_name] = column_data

#list to replace the label for the position of the player
position = []

for i in basketball_data.index:
    
    #gets list of string values of posiiton
    pos = basketball_data['pos'][i]

    #adds the number values for the position into the new list
    if pos == "PG":
        position.append(1)
    elif pos == "SG":
        position.append(2)
    elif pos == "SF":
        position.append(3)
    elif pos == "PF":
        position.append(4)
    elif pos == "C":
        position.append(5)
    else:
        position.append(6)

#replace the column of strings with the number values,
#this is so the computer is able to understand the 
#relation between the player types and postions
basketball_data['pos'] = position

#gets max value to get the column to be a relational propotion of the numbers counted
max_g = basketball_data['G'].max()
max_min = basketball_data['Min'].max()

#turns the values to the percentage of the min and games played as compared to others 
#this is for the lock out and the covid seasons because of the different lengths of seasons
#for those examples to even the scape for the logic
basketball_data['G'] = basketball_data['G'] / max_g      
basketball_data['Min'] = basketball_data['Min'] / max_min
basketball_data['Raptor+/-'] = ((basketball_data['Raptor+/-'] - basketball_data['Raptor+/-'].min())/
                                (basketball_data['Raptor+/-'].max() - basketball_data['Raptor+/-'].min()))

#fill null values for clean run
basketball_data = basketball_data.fillna(method='ffill')

#gets all active years and seperates the data by the years within the
#data frame.
data_by_year = []
count_year = basketball_data.year_id.unique()
for i in count_year:
    data_by_year.append(basketball_data[basketball_data.year_id == i])

#gets each team for each year
for i in data_by_year:
    i.append([item.team_id.unique() for item in data_by_year])

#going to hold the average of the difference in prediction to actual
final_predict = [ 0 for i in range(len(data_by_year))]

#for each year within the data
for i in data_by_year:

    #data attributes to analyze
    x = i[['AST%', 'ORB%', 'DRB%', 'STL%', 'BLK%', '2P%', '3P%', 'FT%']]
    
    #splits to workable values based on posiiton
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        x, i['pos'])

    #execution of linear reggression
    regressor = linear_model.LinearRegression()  
    regressor.fit(X_train, y_train)

    #get the model to predict the value
    y_pred = regressor.predict(X_test)
    
    #will hold the "score" of prediction
    prediction_score = []
    counter = 0

    #goes through each year
    for i in data_by_year:

        #standards for the data
        x_tester = i[['AST%', 'ORB%', 'DRB%', 'STL%', 'BLK%', '2P%', '3P%', 'FT%']]
        y_actual = i['pos']
        #predict based on the year
        y_predict = regressor.predict(x_tester)
    
        #adds the score from the prediction and the year
        prediction_score.append(0)
        y_actually = list(y_actual)

        #goes through the actual vs. the test to find the differnece in the calculations
        for j in range(len(y_actual)):
            prediction_score[counter] += (y_actually[j] - y_predict[j])
        prediction_score[counter] /= len(y_actually)
        counter += 1
    
    # loops throuhg the scores to find the averages for each of the years
    for k in range(len(prediction_score)):
        final_predict[k] += (prediction_score[k] / len(data_by_year))

labels = []
#adds the labels to the data for each of the years
for i in range(len(prediction_score)):
    if i%2 == 1:
        labels.append(1977 + i)

#settings for the output figure for the chart
fig = plt.figure(figsize = (fig_size, fig_size))
ax = plt.axes()
plt.plot(final_predict)
ax.set_xticks([i - 1977 for i in labels])
ax.set_xticklabels(labels)

#line to show the x-axis
zeros = []
for i in range(len(final_predict)):
    zeros.append(0)
plt.plot(range(len(zeros)), zeros)
plt.show()
plt.close()