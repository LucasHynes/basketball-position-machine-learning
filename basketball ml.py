import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from matplotlib import pyplot as plt
min_max_scaler = preprocessing.MinMaxScaler()
fig_size = 12

basketball_data = pd.read_csv("nba-data-historical.csv", engine = 'python')

for column_name, column_data in basketball_data.iteritems():
    if '%' in column_name:
        column_data = [i*0.01 for i in column_data]
        basketball_data[column_name] = column_data

position = []
for i in basketball_data.index:
    pos = basketball_data['pos'][i]

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

basketball_data['pos'] = position


max_g = basketball_data['G'].max()
max_min = basketball_data['Min'].max()

basketball_data['G'] = basketball_data['G'] / max_g      
basketball_data['Min'] = basketball_data['Min'] / max_min
basketball_data['Raptor+/-'] = ((basketball_data['Raptor+/-'] - basketball_data['Raptor+/-'].min())/
                                (basketball_data['Raptor+/-'].max() - basketball_data['Raptor+/-'].min()))

basketball_data = basketball_data.fillna(method='ffill')
  
data_by_year = []

count_year = basketball_data.year_id.unique()

for i in count_year:
    data_by_year.append(basketball_data[basketball_data.year_id == i])


for i in data_by_year:
    i.append([item.team_id.unique() for item in data_by_year])

final_predict = [ 0 for i in range(len(data_by_year))]

for i in data_by_year:
    x = i[['AST%', 'ORB%', 'DRB%', 'STL%', 'BLK%', '2P%', '3P%', 'FT%']]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        x, i['pos'])

    regressor = linear_model.LinearRegression()  
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    
    prediction_score = []
    counter = 0

    for i in data_by_year:
        x_tester = i[['AST%', 'ORB%', 'DRB%', 'STL%', 'BLK%', '2P%', '3P%', 'FT%']]
        y_actual = i['pos']
        y_predict = regressor.predict(x_tester)
    
        prediction_score.append(0)
        y_actually = list(y_actual)
        for j in range(len(y_actual)):
            prediction_score[counter] += (y_actually[j] - y_predict[j])
        prediction_score[counter] /= len(y_actually)
        counter += 1
        
    for k in range(len(prediction_score)):
        final_predict[k] += (prediction_score[k] / len(data_by_year))

labels = []
for i in range(len(prediction_score)):
    if i%2 == 1:
        labels.append(1977 + i)

fig = plt.figure(figsize = (fig_size, fig_size))
ax = plt.axes()
plt.plot(final_predict)
ax.set_xticks([i - 1977 for i in labels])
ax.set_xticklabels(labels)

zeros = []
for i in range(len(final_predict)):
    zeros.append(0)
plt.plot(range(len(zeros)), zeros)
plt.show()
plt.close()