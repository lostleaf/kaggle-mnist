import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cross_validation import cross_val_score

def main():
    with open("train_hog.npy", "rb") as fin:
        train_features = np.load(fin)
    with open("test_hog.npy", "rb") as fin:
        test_features = np.load(fin)
    with open("train.npz", "rb") as fin:
        train_labels = np.load(fin)["train_labels"]
    model = RF(n_estimators=20, max_features=0.7, n_jobs=-1)
    model.fit(train_features, train_labels)
    scores = cross_val_score(model, train_features, train_labels, cv=5)
    print np.mean(scores), np.std(scores)
    # test_pred = model.predict(test_features)
    # result = np.vstack((np.arange(1, 28000 + 1), test_pred)).T
    # print result.shape
    # np.savetxt("result.csv", result, fmt="%d", delimiter=",", header="ImageId,Label", comments='')
    

if __name__ == "__main__":
    main()
