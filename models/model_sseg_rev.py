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
        local_embeddings = self.local_embedder(group_features)
        group_embeddings = local_embeddings.max(dim=-1)[0]
        graph_features = self.graph_embedder(group_coordinates, group_embeddings)   # output: 1, F2, M
        scores = self.classifier(graph_features.unsqueeze(-1)).squeeze(-1)

        return scores, graph_features, local_embeddings


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
            h = SingleHierarchy(**hierachy_params, classifier_dimensions=classifier_dimensions)
            self.hierachies.append(h)
        self.hierachies = nn.ModuleList(self.hierachies)

        embed_dimension = hierarchy_config[0]["dimensions"][1]
        classifier_dimensions = [embed_dimension] + classifier_dimensions

        # Point classifier
        self.pointwise_classifier = PointClassifier(classifier_dimensions)

        self.first_level = hierarchy_config[0]["h_level"]
        self.labelweights = None
        self.strategy = "hard"

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

        feat_list = []
        scores_list = []

        for hierarchy in self.hierachies[:2]:
            # group_coordinates = octree[hierarchy.level][0][:3]
            grouping_indices = octree[hierarchy.level][1]
            group_bboxes = octree[hierarchy.level][2]
            group_origins = group_bboxes.reshape(-1, 3, 2).mean(axis=-1).transpose(1, 0).unsqueeze(0)

            # print(f"\
            #     \tLevel: {hierarchy.level}\n\
            #     \tGroup count: {group_origins.shape}\
            # ", flush=True)

            if hierarchy.level == self.first_level:
                group_features = torch.cat([X[:, :, g_i].unsqueeze(-2) for g_i in grouping_indices], dim=-2)    # shape=(6, N1, G1)
                group_features[0, :3] -= group_origins[0].unsqueeze(-1).repeat(1, 1, group_features.shape[-1])
            else:
                group_features = torch.cat([feat_list[-1][:, :, g_i].unsqueeze(-2) for g_i in grouping_indices], dim=-2) # shape (32, N2, G2)

            # RUN HIERARCHY
            h_scores, h_features, local_features = hierarchy(group_origins, group_features)

            # First level exception
            if hierarchy.level == self.first_level:
                raw_scores = self.pointwise_classifier(local_features)

                point_scores = torch.zeros((1, 14, X.shape[-1]), device=X.device)
                for i, g_i in enumerate(grouping_indices):
                    point_scores[:, :, g_i] = raw_scores[:, :, i]
                scores_list.append(point_scores)

            feat_list.append(h_features)
            scores_list.append(h_scores)

        return scores_list

    def get_loss_depr(self, logits, target, meta=None):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.labelweights)

    def get_loss(self, logits, target, meta=None, get_logits=False):
        # 1. multi labels at higher levels: [0, 1, 1, 0, 1, .... 0]
            # Hamming loss
            # Multilabel hinge loss
        # 2. soft labels are higher levels: [0.6, 0.3, 0.001, 0.002, ... 0.001]


        batch_size = target.shape[0]
        target = target.view(-1)
        target_1h = torch.zeros(target.shape+(14,), dtype=torch.float64, requires_grad=True).to(device=target.device)
        target_1h.scatter_(1, target.view(-1, 1), 1)
        target_1h = target_1h.view(batch_size, -1, 14)
        targets_1h = [target_1h]
        # TODO: meta[0] only for batch_size == 1
        meta = meta[0]
        for lev in sorted(meta.keys())[::-1]:
            grouping_indices = meta[lev][1]
            group_targets = [targets_1h[-1][..., g_i, :].sum(-2, keepdim=True) for g_i in grouping_indices]
            target_1h = torch.cat(group_targets, dim=-2)
            targets_1h.append(target_1h)

        losses = []
        log_probs = []
        for l, t in zip(logits, targets_1h):

            if self.strategy == "soft":
               t = t / t.sum(-1, keepdim=True)
            #    t = torch.max(t, torch.tensor(0, dtype=torch.float64, requires_grad=True).to(t.device))
            elif self.strategy == "hard":
               t = (t > 0.5)

            # sm = torch.nn.functional.softmax(l.transpose(2,1), dim=-1)
            # loss = torch.nn.functional.binary_cross_entropy(sm, target=t, weight=self.labelweights)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(l.transpose(2, 1), t.double(), weight=None)
            losses.append(loss)
            log_probs.append(torch.nn.functional.sigmoid(l.transpose(2, 1)))

        loss = losses[0] + losses[1] + losses[2]

        if not get_logits:
            return loss, None

        # Looped verison of above code
        def to_upper(tg, target_size, grouping):
            upper_logits = torch.zeros((1, target_size, 14), device=tg.device)
            for i, g_i in enumerate(grouping):
                upper_logits[:, g_i] = tg[:, i:i+1].repeat(1, len(g_i), 1)
            return upper_logits

        levels_dict = {1: 5, 2: 3}
        mapped_logits = []
        for l in np.arange(len(log_probs)):
            upper_logits = log_probs[l]
            for lev in np.arange(l, 0, -1):
                upper_logits = to_upper(upper_logits, target_size=log_probs[lev-1].shape[-2], grouping=meta[levels_dict[lev]][1])
            mapped_logits.append(upper_logits)

        prod_logits = torch.prod(torch.cat([ml.unsqueeze(-1) for ml in mapped_logits], dim=-1), dim=-1).transpose(2,1)

        return loss, prod_logits
