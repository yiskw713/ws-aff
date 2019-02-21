import torch
import torch.nn as nn
import torch.nn.functional as F


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


""" Class Activation Mapping """


class CAM(object):
    def __init__(self, model, target_layer_obj, target_layer_aff):
        """
        Args:
            model: ResNet_linear()
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer_obj = target_layer_obj
        self.target_layer_aff = target_layer_aff

        # save values of activations and gradients in target_layer
        self.values_obj = SaveValues(self.target_layer_obj)
        self.values_aff = SaveValues(self.target_layer_aff)

    def forward(self, x, obj_label=None, aff_label=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            label: true label. shape => (1, num_classes)
        Return:
            heatmap: class activation mappings of predicted classes
                    [{obj_id: cam}, {aff_id1: cam1, aff_id2: cam2, ...}]
        """

        pred_obj, pred_aff = self.model(x)

        # object classification
        if obj_label is None:
            pred_obj = torch.sigmoid(pred_obj)
            pred_obj[pred_obj > 0.5] = 1
            pred_obj[pred_obj <= 0.5] = 0
            obj_label = pred_obj
            print("predicted object ids {}".format(pred_obj))

        # affordance classification
        if aff_label is None:
            pred_aff = torch.sigmoid(pred_aff)
            pred_aff[pred_aff > 0.5] = 1
            pred_aff[pred_aff <= 0.5] = 0
            aff_label = pred_aff
            print("predicted affordance ids {}".format(pred_aff))

        weight_fc_obj = list(
            self.model._modules.get('obj_fc').parameters())[0].to('cpu').data
        weight_fc_aff = list(
            self.model._modules.get('aff_fc').parameters())[0].to('cpu').data

        cams_obj = dict()
        cams_aff = dict()

        for i in obj_label.nonzero():
            cam = self.getCAM(self.values_obj, weight_fc_obj, i)
            cams_obj[i[1].item()] = cam    # i[i] is object id

        for i in aff_label.nonzero():
            cam = self.getCAM(self.values_aff, weight_fc_aff, i)
            cams_aff[i[1].item()] = cam.data    # i[i] is affordance id

        return cams_obj, cams_aff

    def __call__(self, x, obj_label=None, aff_label=None):
        return self.forward(x, obj_label, aff_label)

    def getCAM(self, values, weight_fc, index):
        '''
        values: the activations and gradients of target_layer
        activations: feature map before GAP.  shape => (N, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        cam: class activation map.  shape=> (N, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        max_wo_bg, _ = torch.max(cam, dim=1)
        cam[:, 0, :, :] = - max_wo_bg

        cam = cam[index[0], index[1], :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        cam = torch.where(cam > 0.9, cam, torch.tensor([0.]))

        return cam.data

    def get_label(self, x, obj_label, aff_label):

        _, _, H, W = x.shape
        pred_obj, pred_aff = self.model(x)

        weight_fc_obj = list(
            self.model._modules.get('obj_fc').parameters())[0].to('cpu').data
        weight_fc_aff = list(
            self.model._modules.get('aff_fc').parameters())[0].to('cpu').data

        cam_obj = F.conv2d(
            self.values_obj.activations, weight=weight_fc_obj[:, :, None, None])
        cam_aff = F.conv2d(
            self.values_aff.activations, weight=weight_fc_aff[:, :, None, None])

        # resize
        cam_obj = F.interpolate(
            cam_obj, (H, W), mode='bilinear').view(-1, H, W)
        cam_aff = F.interpolate(
            cam_aff, (H, W), mode='bilinear').view(-1, H, W)

        # make object label
        max_cam, _ = torch.max(cam_obj, dim=0)
        cam_obj[0, :, :] = - max_cam    # replace background logit

        for i in obj_label.nonzero():
            cam_obj[i[1]] -= torch.min(cam_obj[i[1]])
            cam_obj[i[1]] /= torch.max(cam_obj[i[1]])
        for i in (0 == obj_label).nonzero():
            cam_obj[i[1]] = 0.

        cam_obj = torch.where(cam_obj > 0.9, cam_obj, torch.tensor([0.]))
        val, index = torch.max(cam_obj, dim=0)
        cam_label_obj = torch.where(
            val > 0.9, index, torch.tensor([-100])).long()

        # make object label
        max_cam, _ = torch.max(cam_aff, dim=0)
        cam_aff[0, :, :] = - max_cam    # replace background logit

        for i in aff_label.nonzero():
            cam_aff[i[1]] -= torch.min(cam_aff[i][1])
            cam_aff[i[1]] /= torch.max(cam_aff[i][1])
        for i in (0 == aff_label).nonzero():
            cam_aff[i[1]] = 0.

        cam_aff = torch.where(cam_aff > 0.9, cam_aff, torch.tensor([0.]))
        val, index = torch.max(cam_aff, dim=0)
        cam_label_aff = torch.where(
            val > 0.9, index, torch.tensor([-100])).long()

        return cam_label_obj, cam_label_aff


""" Grad CAM """


class GradCAM(CAM):
    """
    Args:
        model: ResNet_linear()
        target_layer: conv_layer before Global Average Pooling
    """

    def __init__(self, model, target_layer_obj, target_layer_aff):
        super().__init__(model, target_layer_obj, target_layer_aff)

    def forward(self, x, obj_label=None, aff_label=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
                    [{obj_id: cam}, {aff_id1: cam1, aff_id2: cam2, ...}]
        """

        score_obj, score_aff = self.model(x)

        # object classification
        if obj_label is None:
            pred_obj = torch.sigmoid(score_obj)
            pred_obj[pred_obj > 0.5] = 1
            pred_obj[pred_obj <= 0.5] = 0
            obj_label = pred_obj
            print("predicted object ids {}".format(pred_obj))

        # affordance classification
        if aff_label is None:
            pred_aff = torch.sigmoid(score_aff)
            pred_aff[pred_aff > 0.5] = 1
            pred_aff[pred_aff <= 0.5] = 0
            aff_label = pred_aff
            print("predicted affordance ids {}".format(pred_aff))

        cams_obj = dict()
        cams_aff = dict()

        # caluculate cam of each predicted class
        for i in obj_label.nonzero():
            cam = self.getGradCAM(self.values_obj, score_obj, i)
            cams_obj[i[1].item()] = cam

        for i in aff_label.nonzero():
            cam = self.getGradCAM(self.values_aff, score_aff, i)
            cams_aff[i[1].item()] = cam

        return cams_obj, cams_aff

    def __call__(self, x, obj_label=None, aff_label=None):
        return self.forward(x, obj_label, aff_label)

    def getGradCAM(self, values, score, index):
        self.model.zero_grad()
        score[index[0], index[1]].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data
