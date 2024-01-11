from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split out testing and training datasets
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.2, random_state=1)

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
