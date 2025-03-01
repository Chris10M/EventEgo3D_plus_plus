import torch
import torch.nn as nn
EPS = 1.1920929e-07


class HeatMapJointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(HeatMapJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
                        
            if self.use_target_weight:
                tw = target_weight[:, idx]
                    
                loss_j = self.criterion(
                    heatmap_pred.mul(tw),
                    heatmap_gt.mul(tw)
                )
            else:
                loss_j = self.criterion(heatmap_pred, heatmap_gt)

            loss += loss_j
            
        return loss.mean() / num_joints


class J3dMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(J3dMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):        
        num_joints = output.size(1)

        cnt = 0
        loss = 0
        for idx in range(num_joints):
            j3d_pred = output[:, idx]
            j3d_gt = target[:, idx]
            
            if self.use_target_weight:
                tw = target_weight[:, idx]
                
                loss += self.criterion(
                    j3d_pred.mul(tw),
                    j3d_gt.mul(tw)
                )
            else:
                loss += self.criterion(j3d_pred, j3d_gt)

        return loss.mean() / num_joints


class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, output, target, weight=None):
        if weight is None:
            return self.loss(output, target)
        else:
            weight = weight.view(-1, 1, 1, 1)
            return self.loss(output * weight, target * weight)
        
    
class BoneLoss(nn.Module):
    heatmap_sequence = ["Head", # 0
                        "Neck", # 1
                        "Right_shoulder", # 2 
                        "Right_elbow", # 3
                        "Right_wrist", # 4
                        "Left_shoulder", # 5
                        "Left_elbow", # 6
                        "Left_wrist", # 7
                        "Right_hip", # 8
                        "Right_knee", # 9
                        "Right_ankle", # 10
                        "Right_foot", # 11
                        "Left_hip", # 12 
                        "Left_knee", # 13
                        "Left_ankle", #14
                        "Left_foot" # 15
                        ] 

                        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    kinematic_parents = [ 0, 0, 1, 2, 3, 1, 5, 6, 2, 8,  9, 10,  5, 12, 13, 14]

    print('Kinematic Parents:')
    for i in range(len(heatmap_sequence)):
        print(f'{heatmap_sequence[i]} -> {heatmap_sequence[kinematic_parents[i]]}')

    def __init__(self):
        super(BoneLoss, self).__init__()

        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, pose_predicted, pose_gt, ange_weight, length_weight):
        predicted_bone_vector = pose_predicted - pose_predicted[:, self.kinematic_parents, :]
        gt_bone_vector = pose_gt - pose_gt[:, self.kinematic_parents, :]

        predicted_bone_vector = predicted_bone_vector[:, 1:, :]
        gt_bone_vector = gt_bone_vector[:, 1:, :]

        ange_weight = ange_weight.squeeze(-1)
        ange_weight = torch.mean(ange_weight, dim=1)

        length_weight = length_weight.squeeze(-1)
        length_weight = torch.mean(length_weight, dim=1)

        cos_loss = 1 - self.cos_sim(predicted_bone_vector, gt_bone_vector)
        cos_loss = torch.sum(cos_loss, dim=1)

        cos_loss = torch.mean(cos_loss * ange_weight, dim=0)

        predicted_bone_length = torch.norm(predicted_bone_vector, dim=-1)
        gt_bone_length = torch.norm(gt_bone_vector, dim=-1)
        
        bone_length_loss = torch.sum((predicted_bone_length - gt_bone_length)**2, dim=1)
        bone_length_loss = torch.mean(bone_length_loss * length_weight, dim=0)

        return cos_loss, bone_length_loss
