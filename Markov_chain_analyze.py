import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def str2int(str):
    return int(str)


def splitStr2int(string):
    newStr = string.split(';')
    intArr = toInt(newStr)
    return intArr


def devideBySum(element):
    if element == 0:
        return 0.0
    return 1/element


invVectorValues = np.vectorize(devideBySum)
toInt = np.vectorize(str2int)


def getAllStates(data):  # нет 31 и 63
    arrStates = np.concatenate((data), axis=0)
    return list(set(arrStates))


def calcTransitMatrix(series, states):
    matrixP = np.full((len(states), len(states)), 0e-6)
    for i in range(1, len(series)):
        matrixP[states.index(series[i-1]), states.index(series[i])] += 1

    vectorSum = invVectorValues(np.sum(matrixP, axis=1))
    tmp = np.transpose(np.full(np.shape(matrixP), vectorSum))
    return matrixP*tmp


def confidenceInterval(series, transitMatrix, window, states):
    Psequence = MarkovsChainProbabolities(series, transitMatrix, window, states)
    return np.min(Psequence), np.max(Psequence)


def MarkovsChainProbabolities(series, transitMatrix, window, states):
    Psequence = []
    if len(series) > window:
        for i in range(0, (len(series) - window)):
            seriesInWindow = series[i: i + window]
            Psequence.append(seriesProbabolity(seriesInWindow, transitMatrix, states))
    else:
        Psequence.append(seriesProbabolity(series, transitMatrix, states))
    return Psequence


def seriesProbabolity(series, transitMatrix, states):
    p = 1
    for i in range(1, len(series)):
        p *= transitMatrix[states.index(series[i-1]), states.index(series[i])]
    return p


def checkAnomaliesInSeries(series, transitMatrix, window, interval, states):
    Psequence = MarkovsChainProbabolities(series, transitMatrix, window, states)
    for i in range(0, len(Psequence)):
        if (interval[0] > Psequence[i]) | (Psequence[i] > interval[1]):
            return 1
    return 0


def findAnomalies(series, transitMatrix, window, interval, states):
    Psequence = MarkovsChainProbabolities(series, transitMatrix, window, states)
    result = []
    for i in range(0, len(Psequence)):
        if (interval[0] > Psequence[i]) | (Psequence[i] > interval[1]):
            result.append(i)
    # x = np.arange(0, len(Psequence), 1)
    # minP = np.ones_like(Psequence) * interval[0]
    # maxP = np.ones_like(Psequence) * interval[1]
    # plt.xlim(0, len(Psequence))
    # plt.plot(x, Psequence)
    # plt.plot(x, minP)
    # plt.plot(x, maxP)
    # plt.show()
    if len(result) < 1:
        return "No anomalies"
    return result


if __name__ == '__main__':
    data = pd.read_csv("C:\\Users\\Никита\\Desktop\\data.txt", sep=':')
    dataTrue = pd.read_csv("C:\\Users\\Никита\\Desktop\\data_true.txt", sep=':')
    dataFake = pd.read_csv("C:\\Users\\Никита\\Desktop\\data_fake.txt", sep=':')

    for i in range(0, len(data)):
        data["values"][i] = splitStr2int(data["values"][i])
        dataTrue["true values"][i] = splitStr2int(dataTrue["true values"][i])
        dataFake["fake values"][i] = splitStr2int(dataFake["fake values"][i])

    data = data.merge(dataTrue, how='left', on='users')
    data = data.merge(dataFake, how='left', on='users')

    dataStates = getAllStates(data["values"])

    window = 5
    anomalySeries = []
    trueResult = []
    fakeResult = []
    fakeLocal = []

    for i, timeSeries in data.iterrows():
        MatP = calcTransitMatrix(timeSeries["values"], dataStates)
        interval = confidenceInterval(timeSeries["values"], MatP, window, dataStates)
        trueResult.append(checkAnomaliesInSeries(timeSeries["true values"], MatP, window, interval, dataStates))
        fakeResult.append(checkAnomaliesInSeries(timeSeries["fake values"], MatP, window, interval, dataStates))

    print(f"Anomalies data_true by data train: {100*np.sum(trueResult)/len(trueResult)}%")
    print(f"Anomalies data_fake by data train: {100*np.sum(fakeResult)/len(fakeResult)}%")

    for i, timeSeries in data.iterrows():
        MatP = calcTransitMatrix(timeSeries["true values"], dataStates)
        interval = confidenceInterval(timeSeries["true values"], MatP, window, dataStates)
        if trueResult[i] == 0:
            fakeLocal.append(findAnomalies(timeSeries["values"], MatP, window, interval, dataStates))
        else:
            fakeLocal.append("True data is anomaly")
        anomalySeries.append(checkAnomaliesInSeries(timeSeries["values"], MatP, window, interval, dataStates))

    # print(anomalySeries)
    print(f"\n{100 * (sum(anomalySeries) / len(anomalySeries)): .1f}% of users has anomalies in data")
    print(f"{100 * (sum(fakeResult) / len(fakeResult)): .1f}% of fake data has anomalies in data")
    data.insert(4, "possible positions of anomalies", fakeLocal)
    print(data[["users", "possible positions of anomalies"]])  # .values[:]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
