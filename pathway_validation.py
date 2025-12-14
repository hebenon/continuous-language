import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load("data/pathfinder32_train.npz")
    images, labels = data['images'], data['labels']

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    # Top row: connected, bottom row: not connected
    connected_idx = np.where(labels == 1)[0][:6]
    not_connected_idx = np.where(labels == 0)[0][:6]

    for i, idx in enumerate(connected_idx):
        axes[0, i].imshow(images[idx], cmap='gray')
        axes[0, i].set_title("Connected")
        axes[0, i].axis('off')

    for i, idx in enumerate(not_connected_idx):
        axes[1, i].imshow(images[idx], cmap='gray')
        axes[1, i].set_title("Not Connected")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("pathfinder_check.png", dpi=150)
    print(f"Label distribution: {labels.sum()}/{len(labels)} connected")
    return images, labels


@app.cell
def _(images, labels):
    # Can a simple MLP even solve this?
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_train = images[:5000].reshape(5000, -1)
    y_train = labels[:5000]
    X_test = images[5000:6000].reshape(1000, -1)
    y_test = labels[5000:6000]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    print(f"Logistic Regression: {accuracy_score(y_test, clf.predict(X_test)):.3f}")
    return


if __name__ == "__main__":
    app.run()
