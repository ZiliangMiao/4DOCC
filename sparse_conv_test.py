import MinkowskiEngine as ME
import torch
import torch.nn.functional as F

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
    from utils.deterministic import set_deterministic
    set_deterministic(666)

    # input
    dense_input = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1]])

    # dense_input = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
    #                             [0, 1, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 1, 0, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0]])

    sparse_coords, sparse_feats = to_sparse_coo(dense_input)
    sparse_coords, sparse_feats = ME.utils.sparse_collate(coords=[sparse_coords], feats=[sparse_feats])

    # tensor field and sparse tensor input
    org_tensor_field = ME.TensorField(features=sparse_feats, coordinates=sparse_coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
    sparse_input = org_tensor_field.sparse()

    # sparse convolution (k=3, p=infinite, s=1)
    convk3p1s1 = ME.MinkowskiConvolution(1, 1, kernel_size=3, stride=1, dimension=2, expand_coordinates=False)
    k3s1_sparse_output = convk3p1s1(sparse_input)
    k3s1_out_slice = k3s1_sparse_output.slice(org_tensor_field)

    # sparse convlution (k=3, p=infinite, s=2)
    convk3p1s2 = ME.MinkowskiConvolution(1, 1, kernel_size=3, stride=2, dimension=2)
    k3s2_out = convk3p1s2(sparse_input)
    k3s2_slice_out = k3s2_out.slice(org_tensor_field)

    # get coordinates mapping
    coords_manager = k3s2_out.coordinate_manager
    # coords_key = k3s2_out.coordinate_key
    # tensor_stride = k3s2_out.tensor_stride
    ins, outs = ME.utils.coords.get_coords_map(sparse_input, k3s2_out)
    inc = coords_manager.get_coordinates(1)
    outc = coords_manager.get_coordinates(2)
    for i, o in zip(ins, outs):
        print(f"{i}: ({inc[i]}) -> {o}: ({outc[o]})")

    # transpose convolution TODO: .slice() with stride=2 conv has bug
    convtrk3s2 = ME.MinkowskiConvolutionTranspose(1, 1, kernel_size=3, stride=2, dimension=2)
    trk3s3_out = convtrk3s2(k3s2_out)
    trk3s3_slice_out = trk3s3_out.slice(org_tensor_field)

    # pooling, squeeze dimension
    poolk7s7 = ME.MinkowskiSumPooling(kernel_size=[1, 3], stride=[1, 7], dimension=2)
    poolk7s7_out = poolk7s7(trk3s3_out)

    # dense
    k3s1_dense_out, _, _ = k3s1_sparse_output.dense(shape=torch.Size([1, 1, 7, 7]), min_coordinate=torch.IntTensor([0, 0]))

    # grid sampling, interpolation
    query_pts = torch.tensor([[[(1.5/6-0.5)*2, (1.5/6-0.5)*2]]]).unsqueeze(-2)
    query_feats = F.grid_sample(input=k3s1_dense_out, grid=query_pts, mode='bilinear', padding_mode='zeros',
                                 align_corners=False)

    # polling
    a = 1