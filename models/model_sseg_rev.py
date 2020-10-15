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

def normalize(pc, points_dim=1, length=None, origin=None):
    # pc -= pc.mean(axis=0, keepdims=True)
    # pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
    
    # #TODO: Handle single points
    # if len(pc[0]) == 1:
    #     return pc
    # pc[:3] -= pc[:3].min(axis=points_dim, keepdim=True)[0]
    # if length is None:
    #     length = (pc[:3].max(dim=points_dim)[0] - pc[:3].min(dim=points_dim)[0]).max()
    # pc[:3] = pc[:3]/length - length/2
    pc[:3] -= origin.reshape(3, 1)
    return pc

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
        
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim+3, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(output_dim*2, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.k = k

    def forward(self, coordinates, features):
        
        # TODO: move neigborhood calculation to dataloader
        neigborhood = knn(coordinates, self.k)
        relative_positions = get_graph_feature(coordinates, self.k, idx=neigborhood, diff_only=True)

        features = features.unsqueeze(-1).repeat(1, 1, 1, self.k)
        features_position_added = torch.cat((features, relative_positions), dim=1)

        x = self.conv1(features_position_added)
        x1 = x.mean(dim=-1, keepdim=False)

        x = get_graph_feature(x1, self.k, idx=neigborhood)
        # x = torch.cat((x, relative_positions), dim=1)

        x = self.conv2(x)
        x2 = x.mean(dim=-1, keepdim=False) # x2's size is (batch_size,num_feature,num_points)

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
        self.level = h_level

        input_dim, embed_dim, graph_dim = dimensions
        k_local, k_graph = k

        self.local_embedder = PointNetEmbedder(input_dim, embed_dim, k=k_local)
        self.graph_embedder = GraphEmbedder(embed_dim, graph_dim, k=k_graph)


    def forward(self, group_coordinates, group_features):
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
        group_embeddings = []
        local_embeddings = []
        for group in group_features:                                                 # input:  3, num_points
            local_embedding = self.local_embedder(group.unsqueeze(0))                # output: 1, F, num_points
            local_embeddings += [local_embedding]                                    # append

            group_embedding = F.adaptive_max_pool1d(local_embedding, output_size=1)  # output: 1, F, 1
            group_embeddings += [group_embedding]                                    # append
        group_embeddings = torch.cat(group_embeddings, dim=-1)                       # output: 1, F, M

        # group_embedding: 1, F, M
        # group_coordinates: 1, F, M
        # group_neigborhood: k, M
        group_coordinates = group_coordinates.unsqueeze(0)
        graph_features = self.graph_embedder(group_coordinates, group_embeddings)   # output: 1, F2, M

        return graph_features, local_embeddings


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

        # Point classifier
        # concated_length = np.sum([embed_dimension] + [h["dimensions"][-1] for h in hierarchy_config])
        concated_length = 224
        classifier_dimensions = [concated_length] + classifier_dimensions
        self.point_classifier = PointClassifier(classifier_dimensions)

        self.first_level = hierarchy_config[0]["h_level"]
        self.labelweights = None

    def forward(self, X_batch, octree_batch):
        """
        X: B, 6, N
        octree_batch = [
            {
                5: (pc:(6, M1), groups:(M1,X)),
                1: (pc:(6, M2), groups:(M2,X)),
                1: (pc:(6, 1),  groups:(1,M2)),
            }
        ]
        """
        
        assert len(X_batch) == 1, "Only batch_size=1 is supported"
        octree = octree_batch[0]
        X = X_batch

        pc_list = []
        feat_list = []

        for hierarchy in self.hierachies[:2]:
            group_coordinates = octree[hierarchy.level][0][:3]
            grouping_indices = octree[hierarchy.level][1]
            group_bboxes = octree[hierarchy.level][2]
            group_origins = group_bboxes.reshape(-1, 3, 2).mean(axis=-1)

            if hierarchy.level == self.first_level:
                group_features = [X[0, :, g_i] for g_i in grouping_indices]
                group_features = [normalize(group, origin=origin) for group, origin in zip(group_features, group_origins)]
            else:
                group_features = [feat_list[-1][0, :, g_i] for g_i in grouping_indices]

            # print(hierarchy.level, group_features[0].device, "Min:", np.min([float(e.min()) for e in group_features]), "Max:", np.min([float(e.max()) for e in group_features]),
            # len(group_features), np.min([int(g.shape[-1]) for g in group_features]))

            # RUN HIERARCHY
            h_features, local_embeddings = hierarchy(group_coordinates, group_features)

            # First level exception
            if hierarchy.level == self.first_level:
                point_embeddings = torch.zeros((1, len(local_embeddings[0][0]), X.shape[-1]), device=X.device)
                for i, g_i in enumerate(grouping_indices):
                    point_embeddings[0, :, g_i] = local_embeddings[i]
                feat_list.append(point_embeddings)
                pc_list.append(X[:, :3])


            pc_list.append(group_coordinates.unsqueeze(0))
            feat_list.append(h_features)

        # TODO: Move reflection to dataloader
        concated_features = concat_features(pc_list, feat_list)

        point_features = self.point_classifier(concated_features.unsqueeze(-1)).squeeze(-1)
        
        return point_features

    def forward_old(self, X, octree_batch):
        
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

    def get_loss(self, logits, target):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.labelweights)