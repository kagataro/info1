import datasets
X, Y = datasets.load_linear_example1()

print(f'{X=}')
print(f'{Y=}')

import regression
model = regression.LinearRegression()

print(model.x)

model.fit(X,Y)
print(model.theta)

model.fit(X,Y)
print(model.predict(X))

model.score(X,Y)
print(model.score(X,Y))
