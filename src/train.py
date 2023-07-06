import autokeras as ak
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()

clf = ak.ImageClassifier(max_trials=1, overwrite=True)
clf.fit(x_train, y_train)

loss, accuracy = clf.evaluate(x_test, y_test)
predicted = clf.predict(x_test).flatten().astype('uint8')

print(f'Prediction loss: {loss:.4f}')
print(f'Prediction accuracy: {accuracy:.4f}')
print(classification_report(y_test, predicted))