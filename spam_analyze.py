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
          f"правильно классифицированный спама: {(1 - FPR) * 100: .2f}%\n")
    AUC_ROC = 0.5 * FPR * TPR + 0.5 * (1 - FPR) * (1 + TPR)  # сумма треугольника и трапеции
    print(f"Примерная точность классификации: {AUC_ROC * 100:.2f}%")
    # примерная потому что у нас ROC кривая строится всего по трем точкам
    # (из-за того, что классификатор уже разделяет данные бинарно, а ROC должен сначала разделить классы по порогу)

    fig = plt.figure(figsize=(7, 7))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, FPR, 1], [0, TPR, 1], "b-")
    plt.plot([0, 1], [0, 1], "r-")
    plt.legend(["ROC кривая", "кривая угадывания"])
    plt.show()

    print("Wow!")
