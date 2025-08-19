# @File : models.py
# @Time : 2025/7/2 15:29
# @Author : wyp
# @Purpose :
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel


class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=128)
        self.fc1 = nn.Conv1d(3, 64, 1)
        self.fc2 = nn.Conv1d(64, 128, 1)
        self.fc3 = nn.Conv1d(128, 128, 1)
        self.fc4 = nn.Conv1d(128, 512, 1)
        self.fc5 = nn.Conv1d(512, 2048, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

    def forward(self, input):
        n_pts = input.size()[2]
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)
        outs = []

        out1 = F.relu(self.bn1(self.fc1(xb)))
        outs.append(out1)
        out2 = F.relu(self.bn2(self.fc2(out1)))
        outs.append(out2)
        out3 = F.relu(self.bn3(self.fc3(out2)))
        outs.append(out3)
        matrix128x128 = self.feature_transform(out3)

        out4 = torch.bmm(torch.transpose(out3, 1, 2), matrix128x128).transpose(1, 2)
        outs.append(out4)
        out5 = F.relu(self.bn4(self.fc4(out4)))
        outs.append(out5)

        xb = self.bn5(self.fc5(out5))

        xb = nn.MaxPool1d(xb.size(-1))(xb)
        out6 = nn.Flatten(1)(xb).repeat(n_pts, 1, 1).transpose(0, 2).transpose(0, 1)  # .repeat(1, 1, n_pts)
        outs.append(out6)

        return outs, matrix3x3, matrix128x128


class PointNetSeg(nn.Module):
    def __init__(self,  classes=6, num_instances=200):
        super().__init__()
        self.transform = Transform()

        self.fc1 = nn.Conv1d(3008, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Conv1d(256, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Conv1d(128, classes, 1)
        self.bn4 = nn.BatchNorm1d(classes)

        self.fc4_ins = nn.Conv1d(128, num_instances, 1)
        self.bn4_ins = nn.BatchNorm1d(num_instances)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, input):
        batch_size = input.shape[0]
        inputs, matrix3x3, matrix128x128 = self.transform(input)
        stack = torch.cat(inputs, 1)

        xb = F.relu(self.bn1(self.fc1(stack)))

        xb = F.relu(self.bn2(self.fc2(xb)))

        xb = F.relu(self.bn3(self.fc3(xb)))

        output = F.relu(self.bn4(self.fc4(xb)))
        masks_output = F.relu(self.bn4_ins(self.fc4_ins(xb)))

        # masks_output=torch.round(masks_output)
        # masks_output= masks_output > 0

        masks_output = self.softmax(masks_output)  # B x K x N
        # other_mask_pred = masks_output[:, -1, :]  # B x N

        # x2 = torch.reshape(xb, (batch_size, -1))
        # conf_net = tf.reshape(l3_points, [batch_size, -1])

        # conf_score = self.conf_fc1(x2)
        # conf_score = self.conf_fc2(conf_score)
        # conf_score = self.conf_fc3(conf_score)
        # conf_score = F.sigmoid(conf_score)
        # conf_net = tf_util.fully_connected(conf_net, 256, bn=True, is_training=is_training, scope='conf/fc1', bn_decay=bn_decay)
        # conf_net = tf_util.fully_connected(conf_net, 256, bn=True, is_training=is_training, scope='conf/fc2', bn_decay=bn_decay)
        # conf_net = tf_util.fully_connected(conf_net, num_ins, activation_fn=None, scope='conf/fc3')
        # conf_net = tf.nn.sigmoid(conf_net)

        end_points = {}

        # return self.logsoftmax(output), matrix3x3, matrix128x128
        # return output,masks_output,end_points, other_mask_pred, conf_net, matrix3x3, matrix128x128
        return output, masks_output, end_points, matrix3x3, matrix128x128,xb


class TextEncoderCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", proj_dim=128):
        super(TextEncoderCLIP, self).__init__()
        # CLIP text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        # CLIP  (512) →  (proj_dim)
        self.proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)

    def forward(self, names):
        """
        names: list[str], all classes
        return: (num_classes, proj_dim)
        """
        inputs = self.tokenizer(names, padding=True, return_tensors="pt")
        outputs = self.text_encoder(**inputs)
        # (num_classes, L, D)
        last_hidden = outputs.last_hidden_state
        text_embeds = last_hidden[:, 0, :]  # (num_classes, D)

        text_proj = self.proj(text_embeds)  # (num_classes, proj_dim)
        return text_proj
    
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim) )
    def forward(self, x):
        return self.mlp(x)

class PointTextSegModel(nn.Module):
    def __init__(self, pointnet: nn.Module, text_encoder: nn.Module, proj_dim=128):
        super(PointTextSegModel, self).__init__()
        self.pointnet = pointnet
        self.text_encoder = text_encoder

    
        self.point_proj = Projector(in_dim=128, out_dim=proj_dim)   # PointNetSeg (B,128,N)
        self.text_proj = Projector(in_dim=512, out_dim=proj_dim)    # CLIP : 512 

    def forward(self, points, text_names, tokenizer):
        """
        points: (B, N, 3) 
        text_names: list[str]
        tokenizer: CLIP 的 tokenizer
        """

        # 1. point feature
        outputs, masks_output, end_points, m3x3, m64x64, point_feat = self.pointnet(points.transpose(1,2))
        # point_feat: (B, 128, N)

        # (B, N, 128)
        point_feat = point_feat.transpose(1,2)
        point_proj = self.point_proj(point_feat)  # (B, N, proj_dim)

        # 2. text featue
        text_inputs = tokenizer(text_names, padding=True, return_tensors="pt").to(points.device)
        text_features = self.text_encoder(**text_inputs).last_hidden_state
        text_embeds = text_features[:,0,:]          # (num_classes, 512)
        text_proj = self.text_proj(text_embeds)     # (num_classes, proj_dim)

        # 3. point- text simlariity 
        logits = torch.einsum("bnd,cd->bnc", point_proj, text_proj)  # (B, N, num_classes)

        return logits, outputs, masks_output

