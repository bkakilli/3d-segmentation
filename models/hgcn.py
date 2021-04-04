import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as geom

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


class GraphEmbedder(nn.Module):
    """Graph feature extraction module. Uses Graph Attention Networks to extract graph features
    for each node.
    """
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        
        self.convolution_layers = [
            nn.Sequential(nn.Conv2d(input_dim*2+3, output_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2+3, output_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2+3, output_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2+3, output_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),
        ]
        self.convolution_layers = nn.ModuleList(self.convolution_layers)

    def forward(self, features, edge_vectors):

        num_groups = features.shape[-1]

        x = features
        for layer in self.convolution_layers:
            graph_features_tiled = x[:, :, :1].repeat(1,1,num_groups)
            x = torch.cat((x, x-graph_features_tiled, edge_vectors), dim=1)
            # x = torch.cat((x, relative_positions), dim=1)
            x = layer(x.unsqueeze(-1)).squeeze(-1)
        
        x, _ = x.max(dim=-1, keepdim=False)

        return x


class PointClassifier(nn.Module):
    """Point classifition module for actual segmentation of the points.
    """

    def __init__(self, dims):
        super(PointClassifier, self).__init__()

        self.layers = []
        for i in range(len(dims)-2):
            self.layers += [nn.Conv2d(dims[i], dims[i+1], 1, bias=False)]
            self.layers += [nn.BatchNorm2d(dims[i+1])]
            self.layers += [nn.LeakyReLU(negative_slope=0.1)]
        self.layers += [nn.Conv2d(dims[-2], dims[-1], 1, bias=False)]

        self.classifier = nn.Sequential(*self.layers)
        
    def forward(self, X):
        return self.classifier(X)


class SingleHierarchy(nn.Module):

    def __init__(self, h_level, dimensions, k, classifier_dimensions=None):
        """Init function
        """
        super(SingleHierarchy, self).__init__()
        self.level = h_level

        input_dim, embed_dim, graph_dim = dimensions
        k_local, k_graph = k

        self.local_embedder = PointNetEmbedder(input_dim, embed_dim, k=k_local)
        self.graph_embedder = GraphEmbedder(embed_dim, graph_dim, k=k_graph)

        classifier_dimensions = dimensions[-1:] + classifier_dimensions
        self.classifier = PointClassifier(classifier_dimensions)


    def forward(self, raw_features, edge_vectors):
        """Forward operation of single hierarchy

        Parameters
        ----------
        positions_batch : Tensor
            [BxNx3]. Batch of tensors containing positions of nodes (Point cloud)
        features_batch : Tensor
            [BxFxN]. Batch of tensors of features of the nodes
        groups_batch : list
            [B,]. Groups list dictionary for each input

        Returns
        -------
        (Tensor, Tensor)
            Ouput positions and features of the hierarchy. Can be used as input to the next one.
        """

        """
        First hiearchy will have ~3000 groups
        Second hierarchy will have ~150 groups
        Third hierarchy will have 1 group
        """

        # input: (M, (3, N))
        local_embeddings = self.local_embedder(raw_features) # (B, F, Neighborhood, Points)
        group_embeddings = local_embeddings.max(dim=-1)[0]
        graph_features = self.graph_embedder(group_embeddings, edge_vectors)   # output: 1, F2, M
        # scores = self.classifier(graph_features.unsqueeze(-1)).squeeze(-1)

        return graph_features


class HGCN(nn.Module):
    # TODO: Dense connections accross hierarchies

    def __init__(self,
                 num_classes,
                 aggregation,
                 **kwargs
                ):
        super(HGCN, self).__init__()

        
        hierarchy_config = [
                                # {"h_level": 5, "dimensions": [6, 32, 64], "k": [16, 16]},
                                # {"h_level": 3, "dimensions": [64, 128, 128], "k": [16, 16]},
                                {"h_level": 3, "dimensions": [32, 64, 128], "k": [32, 16]},
                            ]
        input_dim = 6
        classifier_dimensions = [512, num_classes]

        params = hierarchy_config[0]
        self.hierarchy = SingleHierarchy(**params, classifier_dimensions=classifier_dimensions)

        embed_dimension = hierarchy_config[0]["dimensions"][-1]
        classifier_dimensions = [32+embed_dimension] + classifier_dimensions
        
        self.local_embedder = PointNetEmbedder(input_dim, 32)
        # self.graph_embedder = GraphEmbedder(input_dim, 32)

        # Point classifier
        self.pointwise_classifier = PointClassifier(classifier_dimensions)

        self.num_classes = num_classes
        self.first_level = hierarchy_config[0]["h_level"]
        self.labelweights = None
        self.strategy = aggregation

    def forward(self, X_batch):

        X_batch, coordinates = X_batch
        edge_vectors = coordinates[..., :1]-coordinates[..., :]
        num_points_in_group = X_batch.shape[-1]

        local_features = self.local_embedder(X_batch)
        # RUN HIERARCHY
        graph_features = self.hierarchy(local_features, edge_vectors)
        graph_features_tiled = graph_features.unsqueeze(dim=-1).repeat(1, 1, num_points_in_group)

        group_features = local_features[..., 0, :]
        concated_features = torch.cat((group_features, graph_features_tiled), dim=1)

        logits = self.pointwise_classifier(concated_features.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_loss(self, logits, target, meta=None):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.labelweights)
