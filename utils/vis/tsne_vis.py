import numpy as np
import matplotlib.pyplot as plt
import yaml
from nuscenes.nuscenes import NuScenes
from sklearn.manifold import TSNE
from datasets.mos4d.nusc import NuscMosDataset
from datasets.nusc_loader import NuscDataloader
from models.mos4d.models import MosNetwork
from mos4d_baseline_script import load_pretrained_encoder


# def plot_embedding(data, label):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     y_min, y_max = np.min(data,1), np.max(data, 1)
#     data[:, 0] = (data[:, 0] - x_min) / (x_max - x_min)
#     data[:, 1] = (data[:, 1] - y_min) / (y_max - y_min)
#
#     plt.figure()
#     ax = plt.subplot(111)
#     for i in range(data.shape[0]):
#         plt.text(data[i, 0], data[i, 1], str(label[i]),
#                  color=plt.cm.Set1(label[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("TOP Pretrained Features")
#     plt.savefig('TOP Pretrained Features (t-SNE)', dpi=1000)


def get_color(labels):
    colors=["k", "b", "r"]
    tsne_color=[]
    for i in range(len(labels)):
        tsne_color.append(colors[labels[i]])
    return tsne_color


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
    train_set = NuscMosDataset(nusc, cfg['finetune'], dataset_cfg, 'train')
    val_set = NuscMosDataset(nusc, cfg['finetune'], dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, cfg['finetune'], train_set, val_set, True)
    dataloader.setup()
    train_dataloader_list = list(dataloader.train_dataloader())

    # load pretrained encoder
    pretrain_ckpt_path = "/home/ziliang/Projects/4DOCC/logs/ours/moco_51151_5e-5 (v0)/100%nuscenes/vs-0.1_t-3.0_bs-8/version_0/checkpoints/epoch=49.ckpt"
    finetune_model = MosNetwork(cfg['finetune'], dataset_cfg, True)
    finetune_model = load_pretrained_encoder(pretrain_ckpt_path, finetune_model, True)
    encoder = finetune_model.encoder

    # 初始化存储所有数据的列表
    all_curr_pcds = []
    all_mos_labels = []
    all_features = []
    batch_data = train_dataloader_list[0]
    for idx in range(len(batch_data)):
        pcds_4d = batch_data[1][idx]  # [N, 4]
        mos_labels = batch_data[2][idx].detach().numpy()  # [N]
        
        # 获取当前时刻的点云
        ref_time_mask = pcds_4d[:, 3] == 0
        curr_pcd = pcds_4d[ref_time_mask]
        
        # 计算特征
        feats = encoder([pcds_4d]).decomposed_features[0][ref_time_mask].detach().numpy()
        
        # 添加到列表中
        all_curr_pcds.append(curr_pcd)
        all_mos_labels.append(mos_labels)
        all_features.append(feats)
    
    # concat
    curr_pcd = np.concatenate(all_curr_pcds, axis=0)
    mos_labels = np.concatenate(all_mos_labels, axis=0)
    feats = np.concatenate(all_features, axis=0)

    # random downsample
    n_sample = 1000
    sta_idx = np.where(mos_labels == 1)[0]
    mov_idx_ds = np.where(mos_labels == 2)[0]
    indices = np.random.choice(len(sta_idx), size=n_sample, replace=False)
    sta_idx_ds = sta_idx[indices]

    # down-sampled feats and labels
    sta_feats = feats[sta_idx_ds]
    mov_feats = feats[mov_idx_ds]
    sta_labels = mos_labels[sta_idx_ds]
    mov_labels = mos_labels[mov_idx_ds]
    tsne_feats = np.concatenate((sta_feats, mov_feats))
    tsne_labels = np.concatenate((sta_labels, mov_labels))

    #list
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, init='pca', random_state=666, verbose=1)
    tsne_2d_feats = tsne.fit_transform(tsne_feats)

    # plot
    figure = plt.figure(figsize=(15, 15), dpi=100)
    color = get_color(tsne_labels)  # 为6个点配置颜色
    x = tsne_2d_feats[:, 0]  # 横坐标
    y = tsne_2d_feats[:, 1]  # 纵坐标
    plt.scatter(x, y, color=color, s=10)  # 绘制散点图。
    plt.savefig('TOP Pretrained Features (t-SNE)', dpi=100)