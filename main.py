import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


def show_label_dist(label):
    with open("gutenberg_dumas.train.3tuples", "r") as f:
        d = {}
        for line in f.readlines():
            cols = line.strip().split("\t")
            if len(cols) <= 1:
                continue

            l = cols[1]
            act = cols[2]
            if l == label:
                if act in d.keys():
                    d[act] += 1
                else:
                    d[act] = 1

    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)

    labels, values = zip(*sorted_items)
    plt.bar(labels, values)

    plt.title(f"Distribution of action lists for {label}")
    plt.xlabel("Action list")
    plt.ylabel("No. of occurrences")

    plt.xticks(fontsize=5, rotation=45)
    plt.xticks([])
    plt.yscale('log')

    plt.show()


def vectorize(acts, kleene):
    vec = [0 for _ in range(4)]
    if kleene:
        acts_list = acts.split()
        vec[0] = acts_list.count("shift")
        vec[0] += acts_list.count("shift*") * 2

        vec[1] = acts_list.count("left")
        vec[1] += acts_list.count("left*") * 2

        vec[2] = acts_list.count("right")
        vec[2] += acts_list.count("right*") * 2

        vec[3] = acts_list.count("reduce")
        vec[3] += acts_list.count("reduce*") * 2

        return np.array(vec)
    else:
        acts_list = acts.split()
        vec[0] = acts_list.count("shift")
        vec[1] = acts_list.count("left-arc")
        vec[2] = acts_list.count("right-arc")
        vec[3] = acts_list.count("reduce")

        return np.array(vec)


def vector_approach():
    with open("gutenberg_dumas.dev.3tuples_full_seq", "r") as f:
        vecs = []
        labels = []

        for line in f.readlines():
            cols = line.strip().split("\t")
            if len(cols) <= 1:
                continue

            label = cols[1]
            acts = cols[2]

            # Vector with (n_s, n_left, n_right, n_r)
            vec = vectorize(acts, kleene=False)

            # if label == "NOUN" or label == "VERB":
            vecs.append(vec)
            labels.append(label)

        X = np.array(vecs[:1000])
        y = np.array(labels[:1000])
        print(X)

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        unique_labels = list(set(y))
        color_map = plt.get_cmap("tab10")
        colors = {label: color_map(i) for i, label in enumerate(unique_labels)}
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=[colors[label] for label in y], label=y, marker=".", alpha=.1)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=8, label=label)
                   for label in unique_labels]
        plt.legend(handles=handles)

        plt.show()


def normalize_order(acts):
    tokens = acts.split(" ")
    return " ".join(sorted(tokens))
    
def embedding_approach(reduction, n_sentences, order=False):
    with open("gutenberg_dumas.dev.3tuples_full_seq", "r") as f:

        model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = []
        labels = []
        i = 0
        for line in f.readlines():
            cols = line.strip().split("\t")
            if len(cols) <= 1:
                i += 1
                if i == n_sentences:
                    break
                continue

            label = cols[1]
            acts = cols[2]
            if not order:
                acts = normalize_order(acts)

            emb = model.encode(acts)
            embeddings.append(emb)
            labels.append(label)

        X = np.array(embeddings)
        y = np.array(labels)

        if reduction == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
        elif reduction == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)
            X_reduced = tsne.fit_transform(X)
        else:
            raise ValueError("Invalid dimensionality reduction technique")

        unique_labels = list(set(y))
        color_map = plt.get_cmap("tab10")
        colors = {label: color_map(i) for i, label in enumerate(unique_labels)}

        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=[colors[label] for label in y], label=y, marker=".", alpha=.1)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=8, label=label)
                   for label in unique_labels]
        plt.legend(handles=handles)

        plt.title(f"Dimensionality reduction of action list embeddings using {reduction}")
        plt.show()

        return X, X_reduced

        # classes = np.unique(y)
        # num_classes = len(classes)
        # cols = 3  # Max 3 plots per row
        # rows = -(-num_classes // cols)  # Ceiling division to determine the number of rows
        #
        # fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        # axes = np.array(axes).reshape(rows, cols)  # Ensure axes is always 2D
        #
        # for ax, i in zip(axes.flatten(), classes):
        #     ax.scatter(X_reduced[np.where(y == i), 0], X_reduced[np.where(y == i), 1], alpha=0.5, marker=".")
        #     ax.set_title(f"Class {i}")
        #     ax.set_xlabel("X1")
        #     ax.set_ylabel("X2")
        #
        # # Hide unused subplots if any
        # for ax in axes.flatten()[num_classes:]:
        #     ax.set_visible(False)
        #
        # plt.subplots_adjust(hspace=0.4)
        # plt.show()


#vector_approach()
