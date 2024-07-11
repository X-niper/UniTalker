import numpy as np
import torch
import torch.nn as nn


class PCALayer(nn.Module):

    def __init__(self, pca_info_path: str):
        super().__init__()
        pca_info = np.load(pca_info_path)
        feat_to_data = pca_info['components_to_data'].astype(np.float32)
        data_to_feat = feat_to_data.T
        data_bias = pca_info['original_data_mean'].astype(np.float32)
        feat_mean = pca_info['data_components_mean'].astype(np.float32)
        feat_std = pca_info['data_components_std'].astype(np.float32)
        self.register_buffer('feat_to_data', torch.from_numpy(feat_to_data))
        self.register_buffer('data_to_feat', torch.from_numpy(data_to_feat))
        self.register_buffer('data_bias', torch.from_numpy(data_bias))
        self.register_buffer('feat_mean', torch.from_numpy(feat_mean))
        self.register_buffer('feat_std', torch.from_numpy(feat_std))
        self.pca_dim = len(feat_mean)

    def encode(
        self,
        x: torch.Tensor,
        pca_dim: int = None,
    ):
        x = x - self.data_bias
        feat = x @ (self.data_to_feat[:, :pca_dim])
        return feat

    def decode(self, feat: torch.Tensor, pca_dim: int = None):
        data = feat[..., :pca_dim] @ (self.feat_to_data[:pca_dim])
        data = data + self.data_bias
        return data


class PCA:

    def __init__(self, trunc_dim: int = None):
        super().__init__()

    def buld_pca_for_dataset(self, dataset, trunc_dim: int = 1024):
        annot_array = self.load_dataset(dataset)
        self.build_pca(annot_array, trunc_dim)

    def load_dataset(self, dataset):
        annot_list = []
        indices = np.arange(len(dataset))
        for i in indices:
            data_dict = dataset.__getitem__(i)
            annot_list.append(data_dict['data'] - data_dict['template'])
        annot_array = np.concatenate(annot_list, axis=0)
        return annot_array

    def build_incremental_PCA(
        self,
        in_data: np.ndarray,
        trunc_dim=None,
        batch_size=1024,
    ):
        indices = np.arange(len(in_data))
        np.random.shuffle(indices)
        in_data = in_data[indices]
        if in_data.dtype != np.float32:
            in_data = in_data.astype(np.float32)
        from sklearn.decomposition import IncrementalPCA
        ipca = IncrementalPCA(n_components=trunc_dim, batch_size=batch_size)
        data_mean = in_data.mean(0)
        in_data = in_data - data_mean
        in_data_slices = np.split(
            in_data, np.arange(batch_size, len(in_data), batch_size))
        for batch_data in in_data_slices:
            if len(batch_data) != batch_size:
                continue
            ipca.partial_fit(batch_data)

        S = ipca.singular_values_
        components_to_data = ipca.components_
        explained_variance_ratio = ipca.explained_variance_ratio_
        data_components = ipca.transform(in_data)
        data_components_mean = data_components.mean(0)
        data_components_std = data_components.std(0)
        ret_dict = {
            'explained_variance_ratio': explained_variance_ratio,
            'components_to_data': components_to_data,
            'original_data_mean': data_mean,
            'S': S,
            'data_components_mean': data_components_mean,
            'data_components_std': data_components_std
        }
        return ret_dict

    def build_pca(
        self,
        in_data: np.ndarray,
        trunc_dim=None,
        save_compactness=True,
        check_svd=True,
    ):
        ret_dict = {}
        in_data = torch.Tensor(in_data).cuda()
        count, in_dim = in_data.shape
        mean_data = in_data.mean(dim=0)
        in_data_center = in_data - mean_data  # (count, in_dim)
        in_data_center = in_data_center.cpu()
        #  U (count, count), S(count, ), Vt (count, in_dim)
        U, S, Vt = torch.linalg.svd(in_data_center, full_matrices=False)

        if check_svd:
            S_mat = torch.diag(S)
            rebuilt = U @ S_mat @ Vt
            is_precise = torch.allclose(rebuilt, in_data_center)
            print('svd rebuilt all close ?', is_precise)

        var_S = S**2
        if save_compactness:
            compact_vector = var_S / var_S.sum()
            compact_value = compact_vector[0:trunc_dim].sum()
            ret_dict['compactness, ', compact_value]

        pca_bases = Vt[0:trunc_dim].reshape(trunc_dim, -1)
        ret_dict['pca_bases'] = pca_bases
        ret_dict['pca_std'] = S[0:trunc_dim]
        ret_dict['mean_data'] = mean_data
        return ret_dict
