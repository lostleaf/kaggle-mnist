import numpy as np

def main():
    data = np.genfromtxt("train.csv", delimiter=",", skip_header=1, dtype=np.uint8)
    train_data = data[:, 1:]
    train_labels = data[:, 0]
    print np.max(train_labels)
    with open("train.npz", "wb") as fout:
        np.savez(fout, train_data=train_data, train_labels=train_labels)
    data = np.genfromtxt("test.csv", delimiter=",", skip_header=1, dtype=np.uint8)
    with open("test.npy", "wb") as fout:
        np.save(fout, data)

if __name__ == "__main__":
    main()
