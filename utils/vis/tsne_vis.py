import numpy as np
import matplotlib.pyplot as plt
import yaml
from nuscenes import NuScenes
from sklearn import datasets
from sklearn.manifold import TSNE
from datasets.mos4d.nusc import NuscMosDataset


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title("title")
    return fig


if __name__ == '__main__':
    # load config
    with open("../../configs/mos4d.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("../../configs/dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
        dataset_cfg['nuscenes']['root'] = '/home/ziliang' + dataset_cfg['nuscenes']['root']
        dataset_cfg['sekitti']['root'] = '/home/ziliang' + dataset_cfg['sekitti']['root']

    # get data
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, cfg['train'], dataset_cfg, 'train')

    #
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label)
    plt.show()