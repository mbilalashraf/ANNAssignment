from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("Cancer.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:600,0:8]
Y = dataset[:600,8]
x = dataset[600:,0:8]
y = dataset[600:,8]
X
Y
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
import numpy as np
x=np.array([[2,130,79,3,1,42,0.5,80]])
predictions = model.predict(x)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)