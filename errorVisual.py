def getBestValue(int):
  X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=(int/100),random_state=123)
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=100)
  return study.best_value

results=[]
input=[]
for number in range(1,100):
  results.append(getBestValue(number))
  input.append(number)

plt.plot(input,results)
plt.title('Fantasy Football Predictions')
plt.xlabel('Percent of Data Trained')
plt.ylabel('Error Rate Percentage')
plt.show()