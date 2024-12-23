import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_three_latent_distributions(latent_vectors_a, latent_vectors_b, latent_vectors_c, perplexity=50):
    """
    Visualizes the latent space distribution of three sets of latent vectors using t-SNE.

    :param latent_vectors_a: Array of latent vectors for the first distribution.
    :param latent_vectors_b: Array of latent vectors for the second distribution.
    :param latent_vectors_c: Array of latent vectors for the third distribution.
    :param perplexity: Perplexity parameter for t-SNE.
    """
    # Reshape to ensure all arrays are 2D
    latent_vectors_a = latent_vectors_a.reshape(len(latent_vectors_a), -1)
    latent_vectors_b = latent_vectors_b.reshape(len(latent_vectors_b), -1)
    latent_vectors_c = latent_vectors_c.reshape(len(latent_vectors_c), -1)

    # Concatenate all latent vectors
    latent_vectors = np.concatenate([latent_vectors_a, latent_vectors_b, latent_vectors_c], axis=0)

    # Perform t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=12, perplexity=perplexity)
    tsne_results = tsne.fit_transform(latent_vectors)

    # Separate the results
    a_results = tsne_results[:len(latent_vectors_a)]
    b_results = tsne_results[len(latent_vectors_a):len(latent_vectors_a) + len(latent_vectors_b)]
    c_results = tsne_results[len(latent_vectors_a) + len(latent_vectors_b):]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(a_results[:, 0], a_results[:, 1], marker='o', color='tab:blue', alpha=0.6, label='Hexapod Robot with 2 Broken Legs')
    plt.scatter(b_results[:, 0], b_results[:, 1], marker='x', color='tab:red', alpha=0.6, label='Hexapod Robot')
    plt.scatter(c_results[:, 0], c_results[:, 1], marker='s', color='tab:green', alpha=0.6, label='Quadruped Robot')

    # Add legends and labels
    plt.legend(loc='upper right', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=20)
    plt.ylabel('t-SNE Dimension 2', fontsize=20)
    # plt.title('Latent Space Distribution of Three Distributions', fontsize=18)
    plt.show()


# 示例数据
num_samples = 100
latent_dim = 128

# 生成随机数据以模拟两种latent分布
latent_vectors_fl = np.load(f"fl_latent.npy", allow_pickle=True).reshape((100,16))
latent_vectors_sl = np.load(f"sl_latent.npy", allow_pickle=True).reshape((100,16))
latent_vectors_fl2 = np.load(f"fl_latent2.npy", allow_pickle=True)
latent_vectors_fl2 = np.average(latent_vectors_fl2,axis=1).reshape((100,16))


# 可视化
visualize_three_latent_distributions(latent_vectors_fl, latent_vectors_sl,latent_vectors_fl2, perplexity=8)
