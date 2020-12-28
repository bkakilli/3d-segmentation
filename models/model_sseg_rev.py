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


def concat_features(points_h, features_h):

    raw_cloud_b = points_h[0]

    concat_embeddings_b = []    # Do this for every sample in the batch, then concat results
    for i, raw_cloud in enumerate(raw_cloud_b):

        raw_cloud = raw_cloud.transpose(1, 0)
        rc_sum = torch.sum(raw_cloud**2, dim=-1, keepdim=True)
        raw_cloud_size, _ = raw_cloud.shape

        point_embeddings = []   # Do this for each hierarchy
        for groups_b, features_b in zip(points_h, features_h):
            groups, features = groups_b[i], features_b[i]

            group_size = groups.shape[1]

            if group_size == raw_cloud_size:  # Don't waste time/memory if points already correspond to features
                point_embeddings += [features]
            elif group_size == 1:
                point_embeddings += [features.repeat(1, raw_cloud_size)]
            else:
                # When groups in shape (num_points, 3)
                # c = torch.cat((groups, raw_cloud), dim=0)
                # inner = torch.matmul(c, c.transpose(1,0))
                # cc = torch.sum(c**2, dim=-1, keepdim=True)
                # p = cc -2*inner + cc.transpose(1,0)
                # m = p[len(groups):,:len(groups)].argmin(dim=-1)

                # c = torch.cat((groups, raw_cloud), dim=-1)
                # inner = torch.matmul(c.transpose(1,0), c)
                # cc = torch.sum(c**2, dim=0, keepdim=True)
                # p = cc.transpose(1,0) -2*inner + cc

                # _, length = groups.shape
                # m1 = p[length:, :length].argmin(dim=-1)

                groups_sum = torch.sum(groups**2, dim=0, keepdim=True)

                d_square = rc_sum + groups_sum - 2*torch.matmul(raw_cloud, groups)
                m = d_square.argmin(dim=-1)

                point_embeddings += [features[:, m]]

        concat_embeddings_b += [torch.cat(point_embeddings, dim=0).unsqueeze(0)]

    concat_embeddings_b = torch.cat(concat_embeddings_b, dim=0)

    return concat_embeddings_b

class PointNetEmbedder(nn.Module):
    """Local feature extraction module. Uses PointNet to extract features of given point cloud.
    """
    def __init__(self, input_dim, output_dim, k=None):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
                                #    nn.BatchNorm2d(input_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                #    nn.BatchNorm2d(output_dim),
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
            nn.Sequential(nn.Conv2d(input_dim*2, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),
        ]
        self.convolution_layers = nn.ModuleList(self.convolution_layers)

    def forward(self, features):

        num_groups = features.shape[-1]

        x = features
        for layer in self.convolution_layers:
            graph_features_tiled = x[:, :, :1].repeat(1,1,num_groups)
            x = torch.cat((x, x-graph_features_tiled), dim=1)
            # x = torch.cat((x, relative_positions), dim=1)
            x = layer(x.unsqueeze(-1)).squeeze(-1)
        
        x, _ = x.max(dim=-1, keepdim=False)

        return x

class GraphEmbedder_old2(nn.Module):
    """Graph feature extraction module. Uses Graph Attention Networks to extract graph features
    for each node.
    """
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        
        self.convolution_layers = [
            nn.Sequential(nn.Conv2d(input_dim*2+3, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2+3, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2+3, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),

            nn.Sequential(nn.Conv2d(output_dim*2+3, output_dim, kernel_size=1, bias=False),
                        #   nn.BatchNorm2d(output_dim),
                          nn.LeakyReLU(negative_slope=0.1)),
        ]
        self.convolution_layers = nn.ModuleList(self.convolution_layers)

        self.k = k

    def forward(self, features):
        
        x = features
        # TODO: move neighborhood calculation to dataloader
        k = x.shape[-1] if x.shape[-1] < self.k else self.k
        neighborhood = knn(coordinates, k)
        relative_positions = get_graph_feature(coordinates, k, idx=neighborhood, diff_only=True)

        for layer in self.convolution_layers:
            x = get_graph_feature(x, k, idx=neighborhood)
            x = torch.cat((x, relative_positions), dim=1)
            x = layer(x)
            x, _ = x.max(dim=-1, keepdim=False)

        return x
class GraphEmbedder_old(nn.Module):
    """Graph feature extraction module. Uses Graph Attention Networks to extract graph features
    for each node.
    """

    def __init__(self, input_dim, output_dim, k):
        super(GraphEmbedder, self).__init__()
        self.k = k
        self.gat_conv1 = geom.GATConv(input_dim, input_dim*2, bias=True)
        self.gat_conv2 = geom.GATConv(input_dim*2, input_dim*2, bias=True)
        self.gat_conv3 = geom.GATConv(input_dim*2, output_dim, bias=True)

    def forward(self, coordinates, features):

        # Example code:
        # edge_index = torch.tensor([[0, 1, 1, 2],
                                # [1, 0, 2, 1]], dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

        # data = Data(x=x, edge_index=edge_index)
        # >>> Data(edge_index=[2, 4], x=[3, 1])

        neighborhood = knn(coordinates, self.k)
        edge_index = torch.arange(len(neighborhood[0]), requires_grad=False).view(1, -1, 1, 1).repeat(1, 1, self.k, 1).to(device=features.device)
        edge_index_1 = torch.cat((neighborhood.view(*neighborhood.shape, 1), edge_index), axis=-1).view(-1, 2)
        edge_index_2 = torch.cat((edge_index, neighborhood.view(*neighborhood.shape, 1)), axis=-1).view(-1, 2)
        edges = torch.cat((edge_index_1, edge_index_2), axis=0).transpose(1, 0)

        edge_attr = get_graph_feature(coordinates, k=self.k, idx=neighborhood, diff_only=True).view(3, -1).transpose(1, 0)
        edge_attr = torch.cat((edge_attr, -edge_attr), dim=0)
        
        batch_size, _, nodes = features.shape

        features_all = features[0].transpose(1,0)

        gcn = self.gat_conv1(features_all, edge_index=edges, edge_attr=edge_attr)
        gcn = self.gat_conv2(gcn, edge_index=edges, edge_attr=edge_attr)
        gcn = self.gat_conv3(gcn, edge_index=edges, edge_attr=edge_attr)

        gcn = gcn.transpose(1, 0).view(batch_size, -1, nodes)

        return gcn


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


    def forward(self, raw_features):
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
        graph_features = self.graph_embedder(group_embeddings)   # output: 1, F2, M
        # scores = self.classifier(graph_features.unsqueeze(-1)).squeeze(-1)

        return graph_features


class HGCN(nn.Module):
    # TODO: Dense connections accross hierarchies

    def __init__(self,
                 hierarchy_config,
                 input_dim,
                 classifier_dimensions,
                 aggregation,
                 **kwargs
                ):
        super(HGCN, self).__init__()

        params = hierarchy_config[0]
        self.hierarchy = SingleHierarchy(**params, classifier_dimensions=classifier_dimensions)

        embed_dimension = hierarchy_config[0]["dimensions"][-1]
        classifier_dimensions = [32+embed_dimension] + classifier_dimensions
        
        self.local_embedder = PointNetEmbedder(input_dim, 32)
        # self.graph_embedder = GraphEmbedder(input_dim, 32)

        # Point classifier
        self.pointwise_classifier = PointClassifier(classifier_dimensions)

        self.num_classes = classifier_dimensions[-1]
        self.first_level = hierarchy_config[0]["h_level"]
        self.labelweights = None
        self.strategy = aggregation

    def forward(self, X_batch):

        num_points_in_group = X_batch.shape[-1]

        local_features = self.local_embedder(X_batch)
        # RUN HIERARCHY
        graph_features = self.hierarchy(local_features)
        graph_features_tiled = graph_features.unsqueeze(dim=-1).repeat(1, 1, num_points_in_group)

        group_features = local_features[..., 0, :]
        concated_features = torch.cat((group_features, graph_features_tiled), dim=1)

        logits = self.pointwise_classifier(concated_features.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_loss(self, logits, target, meta=None):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.labelweights)
