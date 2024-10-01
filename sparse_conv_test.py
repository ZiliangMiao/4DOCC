import MinkowskiEngine as ME
import numpy as np
import torch


def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)



if __name__ == "__main__":
    dense_input = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1]])

    sparse_coords, sparse_feats = to_sparse_coo(dense_input)
    sparse_coords, sparse_feats = ME.utils.sparse_collate(coords=[sparse_coords], feats=[sparse_feats])

    # tensor field and sparse tensor input
    tensor_field = ME.TensorField(features=sparse_feats, coordinates=sparse_coords,
                                  quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
    sparse_input = tensor_field.sparse()

    # sparse convolution (k=3, p=infinite, s=1)
    convk3p1s1 = ME.MinkowskiConvolution(1, 1, kernel_size=3, stride=1, dimension=2, expand_coordinates=True)
    k3s1_sparse_output = convk3p1s1(sparse_input)
    k3s1_out_slice = k3s1_sparse_output.slice(tensor_field)

    # sparse convlution (k=3, p=infinite, s=2)
    convk3p1s2 = ME.MinkowskiConvolution(1, 1, kernel_size=3, stride=2, dimension=2)
    k3s2_sparse_output = convk3p1s2(sparse_input)

    # TODO: .slice() with stride=2 conv has bug
    k3s2_out_slice = k3s2_sparse_output.slice(tensor_field)

    # dense
    k3s2_dense_out = k3s2_sparse_output.dense(shape=torch.Size([1, 1, 4, 4]), min_coordinate=torch.IntTensor([0, 0]))



    # get strided map
    coords_manager = k3s2_sparse_output.coordinate_manager
    coords_key = k3s2_sparse_output.coordinate_key
    tensor_stride = k3s2_sparse_output.tensor_stride

    ins, outs = ME.utils.coords.get_coords_map(sparse_input, k3s2_sparse_output)
    inc = coords_manager.get_coordinates(1)
    outc = coords_manager.get_coordinates(2)
    for i, o in zip(ins, outs):
        print(f"{i}: ({inc[i]}) -> {o}: ({outc[o]})")

    # polling
    a = 1