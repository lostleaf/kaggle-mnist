import numpy as np
from skimage.feature import hog

def extract(data):
    hog_features = []
    for data_point in data:
        hog_features.append(hog(data_point.reshape(28,28), pixels_per_cell=(7, 7), cells_per_block=(2, 2)))
    return np.array(hog_features)

def main():
    with open("train.npz", "rb") as fin:
        npz = np.load(fin)
        train_data = npz["train_data"]
        train_labels = npz["train_labels"]

    with open("test.npy", "rb") as fin:
        test_data = np.load(fin)
    
    feat_train = extract(train_data)
    feat_test = extract(test_data)

    print feat_train.shape
    print feat_test.shape

    with open("train_hog.npy", "wb") as fout:
        np.save(fout, feat_train)

    with open("test_hog.npy", "wb") as fout:
        np.save(fout, feat_test)

if __name__ == "__main__":
    main()
