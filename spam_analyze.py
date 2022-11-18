import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymorphy2 as pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.naive_bayes import MultinomialNB

morph = pymorphy2.MorphAnalyzer()
vectorizer = CountVectorizer(ngram_range=(2, 2))
classificator = MultinomialNB()


def normalize_and_tokenize(msg):
    no_symbols = re.sub(r'[^\w\s]', ' ', msg).lower()
    token = word_tokenize(no_symbols)
    token = [tk for tk in token if (tk not in stopwords.words('english'))]
    token = [morph.parse(tk)[0].normal_form for tk in token]
    return ' '.join(token)


def get_int_class(string_value):
    if string_value == "spam":
        return 1
    else:
        return 0


def calc_IDF(matrix):
    normal_matrix = copy.copy(matrix)
    N = len(normal_matrix)
    for i in range(0, N):
        normal_matrix[i] = normal_matrix[i].astype(bool)
    DF = np.sum(normal_matrix, axis=0) + 1e-5
    IDF = np.log(N / DF)
    return IDF


def calc_confusion_matrix(fact, predict):
    confusion_matrix = np.array([[0, 0], [0, 0]])
    for i in range(0, len(fact)):
        confusion_matrix[predict[i], fact[i]] += 1
    confusion_matrix = {"TP": confusion_matrix[0, 0], "FP": confusion_matrix[0, 1],
                        "FN": confusion_matrix[1, 0], "TN": confusion_matrix[1, 1]}
    return confusion_matrix


def get_classes(probs, d):
    classes = []
    for p in probs:
        if p[0] >= d:
            classes.append(0)
        else:
            classes.append(1)
    return np.array(classes)


def calc_FPR_TPR(probs, fact_class, step):
    arr_d = np.arange(np.min(probs), np.max(probs) + step, step)
    arr_fpr = []
    arr_tpr = []
    best_d = np.min(probs)
    min_dist = 2.0
    for d in arr_d:
        predicted_classes = get_classes(predicts_prob, d)
        conf_matrix = calc_confusion_matrix(fact_class, predicted_classes)
        arr_fpr.append(conf_matrix["FP"] / (conf_matrix["FP"] + conf_matrix["TN"]))
        arr_tpr.append(conf_matrix["TP"] / (conf_matrix["TP"] + conf_matrix["FN"]))
        dist = np.sqrt(arr_fpr[-1]**2 + (1 - arr_tpr[-1])**2)
        if dist < min_dist:
            best_d = d
            min_dist = dist
    arr_fpr.append(0.0)
    arr_tpr.append(0.0)
    return np.array(arr_fpr), np.array(arr_tpr), best_d


def calc_AUC(x, y):
    S = 0
    for i in range(0, len(x) - 1):
        S += 0.5 * (y[i] + y[i + 1]) * np.abs(x[i + 1] - x[i])
    return S


path = "C:\\Users\\Никита\\Desktop\\Методы проектирования защищенных распределенных систем\\spam.csv"

if __name__ == '__main__':
    data = pd.read_csv(path, sep=",", usecols=[0, 1], dtype={"v1": str, "v2": str}, encoding='latin-1')

    df = pd.DataFrame({"v1": data["v1"], "v2": data["v2"]})
    df["v1"] = df["v1"].apply(get_int_class)
    df["v2"] = df["v2"].apply(normalize_and_tokenize)
    print("End normalize!\n")
    X = df["v2"].values
    y = df["v1"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # матрица отображающая биграммы слов
    # столбцы это возможные пары слов, строки это message
    TF_matrix = vectorizer.fit_transform(X_train).toarray()
    IDF_vector = calc_IDF(TF_matrix)
    TF_IDF = TF_matrix * IDF_vector
    classificator.fit(X=TF_IDF, y=y_train)

    # для просмотра TF_IDF
    # df_out = pd.DataFrame(data=TF_IDF, columns=vectorizer.get_feature_names_out())  # 3900 х ...
    # print(df_out)

    test_TF_matrix = vectorizer.transform(X_test).toarray()
    test_IDF_vector = calc_IDF(test_TF_matrix)
    test_TF_IDF = test_TF_matrix * test_IDF_vector
    predicts = classificator.predict(test_TF_IDF)

    confusion = calc_confusion_matrix(y_test, predicts)
    print(f"confusion matrix:\nTP: {confusion['TP']}\t|\tFP: {confusion['FP']}\n"
          f"FN: {confusion['FN']}\t\t|\tTN: {confusion['TN']}\n")
    TPR = confusion["TP"] / (confusion["TP"] + confusion["FN"])
    FPR = confusion["FP"] / (confusion["FP"] + confusion["TN"])

    print(f"Правильная классификация не спама: {TPR * 100: .2f}%\n"
          f"Неправильно классифицированный спам: {FPR * 100: .2f}%   --->   "
          f"правильно классифицированный спама: {(1 - FPR) * 100: .2f}%")

    predicts_prob = classificator.predict_proba(test_TF_IDF)
    arr_FPR, arr_TPR, border = calc_FPR_TPR(predicts_prob, y_test, 1e-4)
    print(f"Порог классификации: {border:.3f}\n")

    AUC_ROC = 0.5 * FPR * TPR + 0.5 * (1 - FPR) * (1 + TPR)  # сумма треугольника и трапеции
    real_AUC_ROC = calc_AUC(arr_FPR, arr_TPR)
    print(f"Примерная точность классификации: {AUC_ROC * 100:.2f}%")
    print(f"Реальная точность классификации: {real_AUC_ROC * 100: .2f}%")

    fig = plt.figure(figsize=(7, 7))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, FPR, 1], [0, TPR, 1], "r-")
    plt.plot(arr_FPR, arr_TPR, "b-")
    plt.plot([0, 1], [0, 1], c="orange", linestyle="-")
    plt.legend(["ROC кривая классификатора", "реальная ROC кривая", "кривая угадывания"])
    plt.show()

    print("Wow!")
