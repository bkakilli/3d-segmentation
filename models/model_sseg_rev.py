import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom

from svstools import pc_utils


def knn(x, k):
    top=k
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=top, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k, idx=None):
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
                                   nn.BatchNorm2d(input_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(64, output_dim, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(output_dim),
        #                            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):

        features = self.conv1(x.unsqueeze(-1))
        features = self.conv2(features)
        # features = self.conv3(features)
        return features.squeeze(-1)

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
        x = get_graph_feature(x, k=self.k) #here x's size is (batch_size,num_dims(3),num_points,k(k's close points)) 
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # x2's size is (batch_size,num_feature,num_points)

        return x2


class GraphEmbedder(nn.Module):
    """Graph feature extraction module. Uses Graph Attention Networks to extract graph features
    for each node.
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

    def forward(self, groups, features):

        neigborhood = knn(groups, self.k)

        x = get_graph_feature(features, self.k, idx=neigborhood)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.k, idx=neigborhood)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # x2's size is (batch_size,num_feature,num_points)

        return x2

class GraphEmbedder_old(nn.Module):
    """Graph feature extraction module. Uses Graph Attention Networks to extract graph features
    for each node.
    """

    def __init__(self, input_dim, output_dim):
        super(GraphEmbedder, self).__init__()
        self.gat_conv1 = geom.GATConv(input_dim, input_dim*2, bias=True)
        self.gat_conv2 = geom.GATConv(input_dim*2, input_dim*2, bias=True)
        self.gat_conv3 = geom.GATConv(input_dim*2, output_dim, bias=True)

    def forward(self, x):

        # Example code:
        # edge_index = torch.tensor([[0, 1, 1, 2],
                                # [1, 0, 2, 1]], dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

        # data = Data(x=x, edge_index=edge_index)
        # >>> Data(edge_index=[2, 4], x=[3, 1])

        raise NotImplementedError("Implement neigborhood")
        
        batch_size, nodes, emb_dims = x.size()
        A = np.ones((nodes, nodes)) - np.eye(nodes)
        edges = torch.tensor(np.asarray(np.where(A.astype(np.bool))).T, device=x.device)

        # Batch hack
        edges_B = edges.view(1,-1,2).repeat(batch_size, 1, 1)

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1,1,1)*nodes
        edges_B = edges_B + idx_base
        edges_B = edges_B.view(-1, 2).transpose(1,0).contiguous()

        batch = x.view(-1, emb_dims)

        gcn = self.gat_conv1(batch, edge_index=edges_B)
        gcn = self.gat_conv2(gcn, edge_index=edges_B)
        gcn = self.gat_conv3(gcn, edge_index=edges_B)

        raise NotImplementedError("Implement pooling or no pooling")


        gcn = gcn.view(batch_size, nodes, -1)

        gcn = gcn.max(dim=1)[0]

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

    def __init__(self, h_level, dimensions, k):
        """Init function
        """
        super(SingleHierarchy, self).__init__()
        self.h_level = h_level

        input_dim, embed_dim, graph_dim = dimensions

        # LocalEmbedderT = LocalEmbedder if h_level != 5 else PointNetEmbedder
        # self.local_embedder = LocalEmbedderT(input_dim, embed_dim, k=20)
        # self.local_embedder = PointNetEmbedder(input_dim, embed_dim)
        self.local_embedder = LocalEmbedder(input_dim, embed_dim, k=20)

        # No graph embedder for level==1 (which is the global hierarchy with 1 node)
        if h_level > 1:
            # self.graph_embedder = GraphEmbedder(embed_dim, graph_dim)
            self.graph_embedder = GraphEmbedder(embed_dim, graph_dim, k=k)


    def forward(self, features_batch, groups_batch, group_pc_batch):
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

        features_group_batch = []   # Batch processing (manual batching since number of points does not match across differene inputs)
        for features, group_indices, group_pc in zip(features_batch, groups_batch, group_pc_batch):

            # # Get local group embeddings
            # embeddings = self.local_embedder(features.unsqueeze(0))

            # # Octree grouping
            # embeddings_group = [embeddings[..., g_i.view(-1)] for g_i in group_indices]

            # # Octree grouping
            # embeddings_group = [self.local_embedder(features[:, g_i.view(-1)].unsqueeze(0)) for g_i in group_indices]
            embeddings_group = []
            for g_i in group_indices:
                embeddings_group.append(self.local_embedder(features[:, g_i.view(-1)].unsqueeze(0)))
                torch.cuda.empty_cache()

            # Pool local features (symmetric function for permutation invariance)
            embeddings_group = [F.adaptive_max_pool1d(group, output_size=1) for group in embeddings_group]
            embeddings_group = torch.cat(embeddings_group, dim=-1)

            # Get graph features
            if self.h_level > 1:
                features_group = self.graph_embedder(group_pc.unsqueeze(0), embeddings_group)
            else:   # Return local features if there is only 1 node in the group (global hierarchy)
                features_group = embeddings_group

            features_group_batch.append(features_group.squeeze(0))

        return features_group_batch


class HGCN(nn.Module):
    # TODO: Dense connections accross hierarchies

    def __init__(self,
                 hierarchy_config,
                 input_dim,
                 classifier_dimensions,
                 **kwargs
                ):
        super(HGCN, self).__init__()

        self.hierachies = []
        for hierachy_params in hierarchy_config:
            h = SingleHierarchy(**hierachy_params)
            self.hierachies.append(h)
        self.hierachies = nn.ModuleList(self.hierachies)

        # Create initial feature extractor
        embed_dimension = hierarchy_config[0]["dimensions"][0]
        self.raw_embedder = PointNetEmbedder(input_dim, embed_dimension)

        # Point classifier
        concated_length = np.sum([embed_dimension] + [h["dimensions"][-1] for h in hierarchy_config])
        classifier_dimensions = [concated_length] + classifier_dimensions
        self.point_classifier = PointClassifier(classifier_dimensions)


    def forward(self, X, octree_batch):
        
        # Extract low level features
        features = self.raw_embedder(X)

        # Run each hiearchy
        pc_list = [[p[:3, :] for p in X]]
        multi_hier_features = [[f for f in features]]

        for hierarchy in self.hierachies:
            pc_batch = [o[hierarchy.h_level][0][:3, :] for o in octree_batch]
            groups_batch = [o[hierarchy.h_level][1] for o in octree_batch]
            features = multi_hier_features[-1]

            pc_list += [pc_batch]
            multi_hier_features += [hierarchy(features, groups_batch, pc_batch)]

        concated_features = concat_features(pc_list, multi_hier_features)

        point_features = self.point_classifier(concated_features.unsqueeze(-1))
        point_features = point_features.transpose(2,1).contiguous().squeeze(3)
        
        return point_features

    # def to(self, device):
    #     self = super().to(device)
    #     for h in self.hierachies:
    #         h.to(device)
    #     return self
