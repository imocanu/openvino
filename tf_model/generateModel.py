from params import *
from modelBitcoin import *

data = readData(wsize=WSIZE, lookup=LOOKUP, tsize=TSIZE, dfcolumns=DFCOLUMNS, shuffle=True)
m = createModel(dfcolumns=DFCOLUMNS, wsize=WSIZE)
buildModel(data, m, batchSize=BATCH_SIZE, ep=EPOCHS, opt=OPTIMIZER, ls=LOSS)

m.save(os.path.join("saved_model_TF"))

data = readData(wsize=WSIZE, lookup=LOOKUP, tsize=TSIZE, dfcolumns=DFCOLUMNS, shuffle=False)

mse, mae = m.evaluate(data["X_test"], data["y_test"], verbose=1)
mean_absolute_error = data["minmaxScaler"]["Adj Close"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)

predictedPriceNextDay = predictPrice(m, data)
print(f"Predicted price for next day is {predictedPriceNextDay:.2f} $")
print("Accuracy :", getAccuracy(m, data, LOOKUP))
plotFullPrediction(m, data)