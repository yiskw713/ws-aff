import torch
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed


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
            cams_aff[i[1].item()] = cam    # i[i] is affordance id

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

        cam = cam[index[0], index[1], :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        cam = torch.where(cam > 0.8, cam, torch.tensor([0.]))

        return cam.data

    def get_label(self, x, obj_label, aff_label):

        _, _, H, W = x.shape
        pred_obj, pred_aff = self.model(x)
        obj_label = obj_label.view(-1).numpy()
        aff_label = aff_label.view(-1).numpy()

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

        cam_obj, cam_aff = \
            cam_obj.to('cpu').data.numpy(), cam_aff.to('cpu').data.numpy()

        # make object label
        # replace background logit
        cam_bg = np.max(cam_obj, axis=0, keepdims=True)
        cam_bg = - cam_bg
        cam_bg -= np.min(cam_bg)
        cam_bg /= np.max(cam_bg)

        # binarize object label. 0 => bg. 1 => foregrand
        cam_obj = cam_obj[obj_label == 1]
        cam_obj -= np.min(cam_obj)
        cam_obj /= np.max(cam_obj)

        cam_obj = np.concatenate([cam_bg, cam_obj], axis=0)
        cam_obj = np.where(cam_obj > 0.9, cam_obj, 0.)
        val = np.max(cam_obj, axis=0)
        index = np.argmax(cam_obj, axis=0)
        cam_label_obj = np.where(val > 0.9, index, 255).astype(np.uint8)

        # make aff label
        # replace background logits
        cam_bg = - np.max(cam_aff, axis=0, keepdims=True)
        cam_bg -= np.min(cam_bg)
        cam_bg /= np.max(cam_bg)

        cam_aff[aff_label == 0] = 0
        cam_aff[aff_label == 1] -= \
            np.min(cam_aff[aff_label == 1], axis=(1, 2), keepdims=True)
        cam_aff[aff_label == 1] /= \
            np.max(cam_aff[aff_label == 1], axis=(1, 2), keepdims=True)

        cam_aff = np.concatenate([cam_bg, cam_aff], axis=0)
        cam_aff = np.where(cam_aff > 0.9, cam_aff, 0.)
        val = np.max(cam_aff, axis=0)
        index = np.argmax(cam_aff, axis=0)
        cam_label_aff = np.where(val > 0.9, index, 255).astype(np.uint8)

        # supplement each label using background class
        cam_label_obj_ = np.where(
            np.logical_and((cam_label_obj == 255), (cam_label_aff == 0)), 0, cam_label_obj)
        cam_label_obj_ = np.where(
            np.logical_or((cam_label_aff == 0), (cam_label_aff == 255)), cam_label_obj_, 1)
        cam_label_aff_ = np.where(
            np.logical_and((cam_label_obj == 0), (cam_label_aff == 255)), 0, cam_label_aff)

        return cam_label_obj_, cam_label_aff_

    def return_label(self, cam_obj, cam_aff, obj_label, aff_label, i):
        '''
        cam_obj => torch.Tensor. (obj_classes, H, W)
        cam_aff => torch.Tensor. (aff_classes, H, W)
        obj_label => torch.Tensor. (obj_classes, )
        aff_label => torch.Tensor. (aff_classes, )
        '''

        cam_obj, cam_aff = \
            cam_obj.to('cpu').data.numpy(), cam_aff.to('cpu').data.numpy()

        obj_label = obj_label.numpy()
        aff_label = aff_label.numpy()

        # make object label
        # replace background logit
        cam_bg = np.max(cam_obj, axis=0, keepdims=True)
        cam_bg = - cam_bg
        cam_bg -= np.min(cam_bg)
        cam_bg /= np.max(cam_bg)

        # binarize object label. 0 => bg. 1 => foregrand
        cam_obj = cam_obj[obj_label == 1]
        cam_obj -= np.min(cam_obj)
        cam_obj /= np.max(cam_obj)

        cam_obj = np.concatenate([cam_bg, cam_obj], axis=0)
        cam_obj = np.where(cam_obj > 0.9, cam_obj, 0.)
        val = np.max(cam_obj, axis=0)
        index = np.argmax(cam_obj, axis=0)
        cam_label_obj = np.where(val > 0.9, index, 255).astype(np.uint8)

        # make aff label
        # replace background logits
        cam_bg = - np.max(cam_aff, axis=0, keepdims=True)
        cam_bg -= np.min(cam_bg)
        cam_bg /= np.max(cam_bg)

        cam_aff[aff_label == 0] = 0
        cam_aff[aff_label == 1] -= \
            np.min(cam_aff[aff_label == 1], axis=(1, 2), keepdims=True)
        cam_aff[aff_label == 1] /= \
            np.max(cam_aff[aff_label == 1], axis=(1, 2), keepdims=True)

        cam_aff = np.concatenate([cam_bg, cam_aff], axis=0)
        cam_aff = np.where(cam_aff > 0.9, cam_aff, 0.)
        val = np.max(cam_aff, axis=0)
        index = np.argmax(cam_aff, axis=0)
        cam_label_aff = np.where(val > 0.9, index, 255).astype(np.uint8)

        # supplement each label using background class
        cam_label_obj_ = np.where(
            np.logical_and((cam_label_obj == 255), (cam_label_aff == 0)), 0, cam_label_obj)
        cam_label_obj_ = np.where(
            np.logical_or((cam_label_aff == 0), (cam_label_aff == 255)), cam_label_obj_, 1)
        cam_label_aff_ = np.where(
            np.logical_and((cam_label_obj == 0), (cam_label_aff == 255)), 0, cam_label_aff)

        return cam_label_obj_, cam_label_aff_, i

    def parallel_get_label(self, x, obj_label, aff_label):
        '''
        x => torch.Tensor. (N, 3, H, W)
        obj_label => torch.Tensor. (obj_classes, )
        aff_label => torch.Tensor. (aff_classes, )
        '''

        N, _, H, W = x.shape
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
        cam_obj = F.interpolate(cam_obj, (H, W), mode='bilinear')
        cam_aff = F.interpolate(cam_aff, (H, W), mode='bilinear')

        processed = Parallel(n_jobs=-1)(
            [
                delayed(self.return_label)(
                    cam_obj[i], cam_aff[i], obj_label[i], aff_label[i], i)
                for i in range(N)
            ]
        )

        # sort wrt batch number
        processed.sort(key=lambda x: x[2])
        processed_data = [[p[0], p[1]] for p in processed]

        return processed_data
