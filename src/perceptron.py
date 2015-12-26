#!/user/bin/env python
# -*- coding: UTF-8 -*-

import csv as csv
import numpy as np
import pandas as pd
import copy

def clean_data(raw_data):
    raw_data['Gender'] = raw_data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    if len(raw_data.Embarked[raw_data.Embarked.isnull()]) > 0:
        raw_data.loc[raw_data.Embarked.isnull(), 'Embarked'] = raw_data['Embarked'].dropna().mode().values

    ports = list(enumerate(np.unique(raw_data['Embarked'])))
    ports_dict = {name : i for i, name in ports}
    raw_data.Embarked = raw_data.Embarked.map(lambda x: ports_dict[x]).astype(int)

    if len(raw_data.Age[raw_data.Age.isnull()]) > 0:
        raw_data.loc[raw_data.Age.isnull(), 'Age'] = raw_data['Age'].dropna().median()

    raw_data = raw_data.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    return raw_data

def judge(x, w, sign):
    r = 0
    for i in range(len(x)):
        r += x[i] * w[i]
    r += w[len(w) - 1]
    return r * sign > 0

def learn(x, w, sign):
    flag = False
    while (flag == False):
        for i in range(len(x)):
            w[i] += sign * x[i]
        w[len(w) - 1] += sign
        flag = judge(x, w, sign)
    return w


def train(x_s, sign_s):
    w = np.zeros(len(x_s[0]) + 1, float)
    flag = False
    r = 0
    max_bingo = 0
    while (flag == False and r < 100):
        flag = True
        r += 1
        for i in range(len(x_s)):
            x = x_s[i]
            sign = sign_s[i]
            match = judge(x, w, sign)
            if match == False:
                flag = False
                w = learn(x, w, sign)

        bingo = 0
        for i in range(len(x_s)):
            x = x_s[i]
            sign = sign_s[i]
            match = judge(x, w, sign)
            if match:
                bingo += 1

        if bingo > max_bingo:
            final_w = copy.copy(w)
            max_bingo = bingo
            print "Bingo : ", max_bingo, final_w
    print "Final : ", max_bingo, final_w
    if flag:
        print "Success"
    return final_w

def main():
    train_df = pd.read_csv('../data/train.csv', header=0)
    test_df = pd.read_csv('../data/test.csv', header=0)

    train_pd = clean_data(train_df)
    train_pd.loc[train_pd.Survived == 0, 'Survived'] = -1

    test_pd = clean_data(test_df)

    ids = test_df['PassengerId'].values
    train_data = train_pd.values
    test_data = test_pd.values

    x_s = train_data[0::,1::]
    sign_s = train_data[0::,0]
    w = train(train_data[0::,1::], train_data[0::,0])
    bingo = 0
    for i in range(len(x_s)):
        x = x_s[i]
        sign = sign_s[i]
        match = judge(x, w, sign)
        if match:
            bingo += 1
    print "proportion : ", bingo * 100.0 / len(x_s)
    columns = list(test_pd.columns.values)
    columns.append("bias")
    print "Wights : ", zip(columns, w)

    predictions_file = open("perceptron.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['PassengerId', "Survived"])
    for i in range(len(test_data)):
        x = test_data[i]
        r = 0
        for j in range(len(x)):
            r += w[j] * x[j]
        r += w[len(w) - 1]
        if (r > 0):
            survived = 1
        else:
            survived = 0
        open_file_object.writerow([ids[i], survived])
    predictions_file.close()

if __name__ == '__main__':
    main()
