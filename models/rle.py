import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from svstools import pc_utils


def knn(x, k):
    top=k
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=top, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k, idx=None, diff_only=False):
    # x's size is (batch,3,num_points)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        k = x.shape[-1] if x.shape[-1] < k else k
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base #here idx's size is (batch_size,num_points,k)

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size() #num_dims=3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # x's size is (batch_size, num_points, k, num_dims)
    
    if diff_only:
        feature = (feature-x).permute(0, 3, 1, 2) #feature's size is (batch_size,num_dims(3),num_points,k(k's close points))
    else:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2) #feature's size is (batch_size,num_dims(3),num_points,k(k's close points))

    return feature
    

class DGCNN_Embedder(nn.Module):
    def __init__(self, input_dim, emb_dims=1024, k=20, activation=nn.LeakyReLU, activation_params={"negative_slope": 0.2}):
        super(DGCNN_Embedder, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   activation(**activation_params))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   activation(**activation_params))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   activation(**activation_params))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   activation(**activation_params))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   activation(**activation_params))

    def forward(self, x):
        batch_size, num_input_features, num_neighbors, num_points = x.shape

        # Treat all groups as a single point cloud (will rearrange in the end)
        x = x.reshape(batch_size, num_input_features, -1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        point_features = self.conv5(x)
        num_output_features = point_features.shape[1]

        # Rearrange groups
        point_features = point_features.reshape(batch_size, num_output_features, num_neighbors, num_points)
        
        # x1 = F.adaptive_max_pool1d(point_features, 1).view(batch_size, -1, num_neighbors)
        # x2 = F.adaptive_avg_pool1d(point_features, 1).view(batch_size, -1, num_neighbors)
        x1 = torch.max(point_features, dim=-1)[0]
        x2 = torch.max(point_features, dim=-1)[0]
        group_features = torch.cat((x1, x2), 1)

        return point_features[..., 0, :], group_features


class PointNetEmbedder(nn.Module):
    """Local feature extraction module. Uses PointNet to extract features of given point cloud.
    """
    def __init__(self, input_dim, output_dim, k=None):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(input_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(64, output_dim, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(output_dim),
        #                            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):

        features = self.conv1(x)
        features = self.conv2(features)
        # features = self.conv3(features)
        return features

class LocalEmbedder(nn.Module):
    """Local feature extraction module. Uses DGCNN to extract features of given point cloud.
    """
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(output_dim*2, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.k = k

    def forward(self, x):

        # print("shape x:", x.shape)
        # TODO: Fix here with a permanent stuff
        repeat = len(x[0,0]) == 1
        if repeat:
            x = x.repeat(1, 1, 2)
        x = get_graph_feature(x, k=self.k) #here x's size is (batch_size,num_dims(3),num_points,k(k's close points)) 
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # x2's size is (batch_size,num_feature,num_points)

        if repeat:
            x2 = x2[..., :1]

        return x2



class MLP(nn.Module):
    """Generic Multi-Layer Perceptron.
    """

    def __init__(self, dims, activation=nn.ReLU, activation_params={}, bias=False, batch_norm=False):
        super(MLP, self).__init__()

        self.layers = []
        for i in range(len(dims)-2):
            self.layers += [nn.Conv1d(dims[i], dims[i+1], 1, bias=bias)]
            if batch_norm:
                self.layers += [nn.BatchNorm1d(dims[i+1])]
            self.layers += [activation(**activation_params)]
        self.layers += [nn.Conv1d(dims[-2], dims[-1], 1, bias=bias)]

        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, X):
        return self.mlp(X)


"""TRYOUT STACK
#TODO: I am making the LocalEmbedder a PointNet module to get going. Later this must be changed to DGCNN or something else
#TODO: Use HEIGHT of a block in the local context information    
#TODO: Try with different group dimensions (other than 50cm) 
#TODO: Improve point_classifier
#TODO: Verify BatchNorm and LeakyRelu
#TODO: Improve positional_embedder
#TODO: Check the effect of bias in modules
"""

class RLE(nn.Module):

    def __init__(self, num_classes, **kwargs):
        super(RLE, self).__init__()
        
        self.input_dim = 6
        self.local_embedder = DGCNN_Embedder(input_dim=self.input_dim, k=32)

        # Point classifier
        num_point_features = 1024 + 256 # Local context embedding and raw point embedding lengths
        self.point_classifier = MLP([num_point_features, 512, num_classes])

        # Define Transformer modules and parameters
        self.positional_embedder = MLP([4, 32, 256])
        self.WQ = MLP([2048, 256])
        self.WK = MLP([2048, 256])
        self.WV = MLP([2048, 256])
        self.mapping = MLP([256, 256])

        self.num_classes = num_classes
        self.labelweights = None

    def transformer(self, group_embeddings, positional_data):

        query = group_embeddings[:, :, :1]
        batch_size, input_feature_dim, num_neighbors = group_embeddings.shape

        query_E = torch.tile(self.WQ(query), dims=(1, 1,num_neighbors ))
        key_E = self.WK(group_embeddings)
        value_E = self.WK(group_embeddings)

        # Get positional encoding (keys) for the transformer
        positional_encoding = self.positional_embedder(positional_data)
        
        # Vector based attentino
        scaled = torch.softmax(self.mapping(query_E - key_E + positional_encoding), dim=-1)
        aggregated = torch.sum(scaled * value_E + positional_encoding, dim=-1)

        return aggregated

    def forward(self, X):
        """Feed forward function

        Args:
            X (tensor): Group coordinates of a neighborhood.
                              Coordinates are relative to the corresponding group origin.
                              First neighbor is the query group.
                              Shape: (B, input_channels, neighborhood, num_points_in_group)
            edge_vectors (tensor): Relative positions of neighbors w.r.t the query group Shape: (B, 3, neighborhood)
            

        Returns:
            tensor: Logits with shape (B, num_classes, num_points_in_group)
        """
        X, coordinates = X
        # X = X[..., :256]
        edge_vectors = coordinates[..., :1]-coordinates[..., :]
        group_heights = coordinates[:, -1:, :]
        positional_data = torch.cat((edge_vectors, group_heights), dim=1)

        batch_size, num_input_channels, num_neighbors, num_points_in_group = X.shape
        assert self.input_dim == num_input_channels # Sanity check

        # Find pointwise and group embeddings
        point_embeddings, group_embeddings = self.local_embedder(X)     # (B, embed_dim, neighbors)

        # Append some metadata (relative positioning and group height) to group embeddings
        # group_embeddings = torch.cat((group_embeddings, positional_data), dim=1)    # TODO: Is this necessary?

        # Apply self attention to get the local context embedding
        local_context_info = self.transformer(group_embeddings, positional_data)

        # Concatenate local context info with point embeddings
        local_context_info = torch.tile(local_context_info.reshape(batch_size, -1, 1), dims=(1, 1, num_points_in_group))
        point_features = torch.cat((point_embeddings, local_context_info), dim=1)
        
        # Classify points with an MLP
        logits = self.point_classifier(point_features)

        return logits

    def get_loss(self, logits, target, meta=None):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.labelweights)
