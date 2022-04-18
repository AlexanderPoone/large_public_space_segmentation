'''
Segmentation by transformers
Alex Poon 2022-03-15

PyTorch (CUDA version) is required.

Tensorboard is recommended.

Some design is borrowed from Hengshuang Zhao, Li Jiang, Jiaya Jia et al. from the University of Oxford, the University of Hong Kong, and the Chinese University of Hong Kong.

This algorithm uses max pooling, FPS, S3DIS, and PyTorch-geometric instead.
'''
from os.path import dirname, realpath

import torch

from torch.nn import BatchNorm1d, Identity, Linear, Module, ModuleList, ReLU, Sequential
from torch.nn.functional import log_softmax, nll_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from torch_cluster import fps, knn_graph
from torch_geometric.datasets import S3DIS
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.transforms import Compose, NormalizeScale, RandomRotate, RandomTranslate
from torch_geometric.utils import intersection_and_union
from torch_scatter import scatter_max

# Globa; parameters
MODE = 'train'                  # What to do: 'train' or 'test'
TEST_MODE = 'prediction'        # If 'test', what to output to .XYZ ASCII Point Cloud: 'original', 'ground_truth' or 'prediction'

path = f'{dirname(realpath(__file__))}/../data/S3DIS'
print(path)

transform = Compose((
    RandomRotate(9, axis=0),
    RandomRotate(9, axis=1),
    RandomRotate(9, axis=2),
    RandomTranslate(.05)
))
pre_transform = NormalizeScale()
train_dataset = S3DIS(path, test_area=4, train=True, transform=transform, pre_transform=pre_transform)
test_dataset = S3DIS(path, test_area=4, train=False, pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6)

def MLP(channels, batch_norm=True):
    return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), BatchNorm1d(channels[i]) if batch_norm else Identity(), ReLU()) for i in range(1, len(channels))])

class TransitionDown(Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce cardinality and uses an MLP to performs dimensionality augmentation
    '''
    def __init__(self, in_channels, out_channels, ratio=0.1, k=16):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.mlp = MLP((in_channels, out_channels))

    def forward(self, x, pos, batch):
        # Farthest Point Sampling (FPS)
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # Calculate For Each Cluster the K Nearest Points
        sub_batch = batch[id_clusters] if batch is not None else None

        # Beware Of Self Loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # Transformation Of Features Through A Simple MLP
        x = self.mlp(x)

        # Max Pool Onto Each Cluster the Features From KNN In Points
        # scatter_max(): Maximizes all values from the src tensor into out at the indices specified in the index tensor along a given axis dim.If multiple indices reference the same location, their contributions maximize (cf. scatter_add()). the second return tensor contains index location in src of each maximum value (known as argmax).
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0], dim_size=id_clusters.size(0), dim=0)

        # Keep Only the Clusters And Their Max-pooled Features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch

class TransformerBlock(Module):
    '''
        Attention
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_in = Linear(in_channels, in_channels)
        self.linear_out = Linear(out_channels, out_channels)

        self.positional_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attention_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.positional_nn, attn_nn=self.attention_nn)

    def forward(self, x, pos, edge_index):
        x = self.linear_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.linear_out(x).relu()
        return x

class TransitionUp(Module):
    '''
        Performs dimensionality reduction and interpolates back to higher resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP((in_channels, out_channels))
        self.mlp = MLP((out_channels, out_channels))

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # Transforms Low-resolution Features And Reduce the Number Of Features
        x_sub = self.mlp_sub(x_sub)

        # Interpolates Low-resolution Feats To High-resolution Points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3, batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x

class SegmentationGNN(Module):
    '''
        the Combined Network
    '''
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # Dummy Feature is Created If There is None Given
        in_channels = max(in_channels, 1)

        # First Block
        self.mlp_input = MLP((in_channels, dim_model[0]))

        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])

        # Backbone Layers
        self.transformers_up = ModuleList()
        self.transformers_down = ModuleList()
        self.transition_up = ModuleList()
        self.transition_down = ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down Block Followed By A Point Transformer Block
            self.transition_down.append(TransitionDown(in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k))
            self.transformers_down.append(TransformerBlock(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]))

            # Add Transition Up Block Followed By Point Transformer Block
            self.transition_up.append(TransitionUp(in_channels=dim_model[i + 1], out_channels=dim_model[i]))
            self.transformers_up.append(TransformerBlock(in_channels=dim_model[i], out_channels=dim_model[i]))

        # Centre Block Layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)
        self.transformer_summit = TransformerBlock(in_channels=dim_model[-1], out_channels=dim_model[-1])

        # Class Score Calculation
        self.mlp_output = Sequential(Linear(dim_model[0], 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, out_channels))

    def forward(self, x, pos, batch=None):

        # Add Dummy Features In Case There is None
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # First Block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # Save Outputs For Shortcut Connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # Backbone Down: Cardinality reduction + Dimension augmentation
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # Centre Block
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # Backbone Up: Cardinality augmentation + Dimension reduction
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-(1+i)](x=out_x[-(2+i)], x_sub=x, pos=out_pos[-(2+i)], pos_sub=out_pos[-(1+i)], batch_sub=out_batch[-(1+i)], batch=out_batch[-(2+i)])

            edge_index = knn_graph(out_pos[-(2+i)], k=self.k, batch=out_batch[-(2+i)])
            x = self.transformers_up[-(1+i)](x, out_pos[-(2+i)], edge_index)

        # Score of Each Of the 13 Category
        out = self.mlp_output(x)

        return log_softmax(out, dim=-1)

# Tensorboard
writer = SummaryWriter()

device = torch.device('cuda')
model = SegmentationGNN(6, train_dataset.num_classes, dim_model=[32, 64, 128, 256, 512], k=16).to(device)

# Load from Checkpoint
model.load_state_dict(torch.load('transformer_checkpoint.pt'))
optimiser = Adam(model.parameters(), lr=.001)
scheduler = StepLR(optimiser, step_size=40, gamma=.1)

def train():
    model.train()

    total_nodes = 0
    correct_nodes = 0
    total_loss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimiser.zero_grad()
        out = model(data.x, data.pos, data.batch)
        # Negative Log Likelihood Loss
        loss = nll_loss(out, data.y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(train_loader)}] | Train accracy: {correct_nodes / total_nodes:.4f} | Loss: {total_loss / 10:.4f}')
            writer.add_scalar('Accuracy/train', correct_nodes / total_nodes, i+1)
            writer.add_scalar('Loss/train', total_loss / 10, i+1)
            total_nodes = 0
            correct_nodes = 0
            total_loss = 0


def test(loader):
    model.eval()

    preds = []  # Calculate IoU on the full dataset in place of a per-batch basis
    labels = []  # Stack the predictions and labels
    for i, data in enumerate(loader):
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)
        preds.append(pred)
        labels.append(data.y)

    # print('Test bottleneck 1')
    preds_tensor = torch.hstack(preds).type(torch.LongTensor)
    # print('Test bottleneck 2')
    labels_tensor = torch.hstack(labels).type(torch.LongTensor)

    intersection, union = torch.zeros((13)), torch.zeros((13))
    # Intersection_and_union Does One-hot Encoding,
    # Since the Full Label Matrix is Too Large, We Have To Do It Twice

    # print('Test bottleneck 3')
    intersection_sub, union_sub = intersection_and_union(preds_tensor[:7000000], labels_tensor[:7000000], 13)
    # print('Test bottleneck 4')
    intersection += intersection_sub
    union += union_sub

    # print('Test bottleneck 5')
    intersection_sub, union_sub = intersection_and_union(preds_tensor[7000000:], labels_tensor[7000000:], 13)
    # print('Test bottleneck 6')
    intersection += intersection_sub
    union += union_sub

    ious = intersection / union
    print(ious)

    # Calculate mean IoU
    iou = ious[~torch.isnan(ious)].mean()
    return iou

def test_output_visualisation(loader):
    '''
    This function will output either the original point cloud, the segmentation ground truth, or the predictions into .xyz ASCII format.
    '''
    model.eval()

    preds = []  # Calculate IoU on the full dataset in place of a per-batch basis
    # coords = []
    labels = []  # Stack the predictions and labels
    num = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)
        preds.append(pred)
        # print(dir(data))
        print(data.y.shape)
        print(pred.shape)

        for _ in range(6):
            num += 1
            with open(f'pred_{num}.xyz', 'w') as f:
                # batchpos = data.pos[data.batch == _]
                batchx = data.x[data.batch == _]
                batchy = data.y[data.batch == _]
                batchpred = pred[data.batch == _]
                for x in range(batchx.shape[0]):
                    if TEST_MODE == 'original':
                        f.write(f'{batchx[x,3]} {batchx[x,4]} {batchx[x,5]} {batchx[x,0]} {batchx[x,1]} {batchx[x,2]}\n')
                    elif TEST_MODE == 'ground_truth':
                        f.write(f'{batchx[x,3]} {batchx[x,4]} {batchx[x,5]} {batchy[x]}\n')
                    else:           #' prediction'
                        f.write(f'{batchx[x,3]} {batchx[x,4]} {batchx[x,5]} {batchpred[x]}\n')

                #print(torch.hstack([data.pos,data.y]))
                # coords.append(data.pos.cpu().numpy())
                labels.append(data.y)

    # print(coords[:100])
    # print('-------------------------')
    # print(preds[:100])
    # print('~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(labels[:100])

    # print('Test bottleneck 1')
    preds_tensor = torch.hstack(preds).type(torch.LongTensor)
    # print('Test bottleneck 2')
    labels_tensor = torch.hstack(labels).type(torch.LongTensor)

    intersection, union = torch.zeros((13)), torch.zeros((13))
    # Intersection_and_union Does One-hot Encoding,
    # Since the Full Label Matrix is Too Large, We Have To Do It Twice

    # print('Test bottleneck 3')
    print(preds_tensor[0].numpy())
    print(len(preds_tensor))
    intersection_sub, union_sub = intersection_and_union(preds_tensor[:7000000], labels_tensor[:7000000], 13)
    # print('Test bottleneck 4')
    intersection += intersection_sub
    union += union_sub

    # print('Test bottleneck 5')
    intersection_sub, union_sub = intersection_and_union(preds_tensor[7000000:], labels_tensor[7000000:], 13)
    # print('Test bottleneck 6')
    intersection += intersection_sub
    union += union_sub

    ious = intersection / union
    print(ious)

    # Calculate mean IoU
    iou = ious[~torch.isnan(ious)].mean()
    return iou

if __name__ == '__main__':
    if MODE == 'test':
        iou = test_output_visualisation(test_loader)
    else:
        for epoch in range(1, 100):
            train()
            iou = test(test_loader)
            print(f'Epoch: {epoch:03d}, Test IoU: {iou:.4f}')
            # Writer.add_scalar('accuracy/test', Iou, Epoch)
            scheduler.step()
            torch.save(model.state_dict(), 'transformer_checkpoint - Copy (9).pt')