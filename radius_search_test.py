import torch
import torch_cluster.radius as search_radius

print(torch.__version__)
print(torch.version.cuda)

x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
batch_x = torch.tensor([0, 0, 0, 0])
y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
batch_y = torch.tensor([0, 0])
assign_index = search_radius(x, y, 1.5, batch_x, batch_y)

a = 1