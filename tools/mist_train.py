import torch
import torchvision
import numpy as np
from utils.utils_bbox import DecodeBox
from torchvision.transforms import transforms
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)
from nets.frcnn_training import FasterRCNNTrainer
import torch.backends.cudnn as cudnn
from nets.frcnn import FasterRCNN
import os
import cv2
from predict import predict

# 定义数据预处理方法
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



# 定义MIST方法
def mist_train(model, train_dataloader, epoches):
    Cuda = True if torch.cuda.is_available() else False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 14
    
    std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(num_classes + 1)[None].to(device)
    decodebox = DecodeBox(std, num_classes)
    save_dir = 'surtool_model/'
    fp16 = True
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    
    
    # 初始化模型和优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
    # 冻结主干网络
    for param in model.extractor.parameters():
        param.requires_grad = False
    # 冻结bn层
    model.freeze_bn()

    # model_train = model.train()
    # if Cuda:
    #     model_train = torch.nn.DataParallel(model_train)
    #     cudnn.benchmark = True
    #     model_train = model_train.cuda()

    train_util = FasterRCNNTrainer(model, optimizer).to(device)
    for epoch in range(epoches):
        print(f"############################ epoch: {epoch} ############################")
        total_loss = 0
        rpn_loc_loss = 0
        rpn_cls_loss = 0
        roi_loc_loss = 0
        roi_cls_loss = 0
        for iteration, batch in enumerate(train_dataloader):
            flag = True
            # 使用当前模型对batch中的所有图像进行预测
            images = batch['imgs']
            labels = batch['labels'].to(device)
            pseudo_boxes = [] #生成的假box
            pseudo_labels = [] #生成的假标签

            # 对batch中的每副图像预测box
            for image in images:
                pseudo_box = [] #生成的假box
                pseudo_label = [] #生成的假标签
                pseudo_conf = [] #对应的概率
                # ---------------------------------------------------#
                #   计算输入图片的高和宽
                # ---------------------------------------------------#
                image_shape = np.array(np.shape(image)[1:3])
                # ---------------------------------------------------#
                #   计算resize后的图片的大小，resize后的图片短边为600
                # ---------------------------------------------------#
                # input_shape = get_new_img_size(image_shape[0], image_shape[1])
                # print(input_shape)
                # ---------------------------------------------------------#
                #   给原图像进行resize，resize到短边为600的大小上
                # ---------------------------------------------------------#

                # image_data = resize_image(image, [input_shape[1], input_shape[0]])
                # image_data = cv2.resize(image, [input_shape[1], input_shape[0]])
                # 添加上batch_size维度
                image_data  = np.expand_dims(image, 0)
                image_data = torch.from_numpy(image_data).float()
                if Cuda:
                    image_data = image_data.cuda()

                roi_cls_locs, roi_scores, rois, _ = model(image_data)  # 取出roi预测框和概率
                
                # 利用classifier的预测结果对建议框进行解码，获得预测框
                with torch.no_grad():
                    results = decodebox.forward(roi_cls_locs, roi_scores, rois, (720, 1280), image_shape, nms_iou = 0.5, confidence = 0.01)

                if len(results[0]) > 0:
                    # top_label = np.array(results[0][:, 5], dtype='int32') #[0, 3]的类型，代表预测框的编号
                    top_label = results[0][:, 5]
                    top_conf = results[0][:, 4] #[0.97, 0.82] 代表概率
                    top_boxes = results[0][:, :4] #top, left, bottom, right
                    
                    # 改成left, top, right, bottom
                    left, top, right, bottom = top_boxes[:, [1, 0, 3, 2]].T
                    top_boxes = np.array([left, top, right, bottom]).T
                    
                    # 过滤掉两边的黑边
                    top_boxes_ = top_boxes[np.logical_and(top_boxes[:,0]>130, top_boxes[:,2]<1050)]
                    top_conf_ = top_conf[np.logical_and(top_boxes[:,0]>130, top_boxes[:,2]<1050)]
                    top_label_ = top_label[np.logical_and(top_boxes[:,0]>130, top_boxes[:,2]<1050)]
                    # 过滤掉上下的黑边
                    top_boxes = top_boxes_[np.logical_and(top_boxes_[:,1]>50, top_boxes_[:,3]<680)]
                    top_conf = top_conf_[np.logical_and(top_boxes_[:,1]>50, top_boxes_[:,3]<680)]
                    top_label = top_label_[np.logical_and(top_boxes_[:,1]>50, top_boxes_[:,3]<680)]
                    
                else:
                    flag = False
                    top_label = []
                    top_conf = []
                    top_boxes = []
                    #continue
                
                # 取出在label中标注的类
                for i in range(len(top_label)):
                    if top_label[i] in labels:
                        pseudo_box.append(top_boxes[i])
                        pseudo_label = np.append(pseudo_label, top_label[i])
                        pseudo_conf = np.append(pseudo_conf, top_conf[i])
                        
                # 取出概率最高的3个
                #sorted_indices = np.argsort(pseudo_conf)[::-1] # 按概率排序
                #sorted_indices = sorted_indices[:3] # 取概率最高的3个，不满3个取全部
                
                pseudo_box_ = []
                pseudo_label_ = []
                # 取出label中每类概率最高的2个
                unique_labels = np.unique(pseudo_label)
                
                for label in unique_labels:
                    label_indices = np.where(pseudo_label == label)[0]
                    if len(label_indices) > 0:
                        label_boxes = []
                        label_conf = []
                        for i in label_indices:
                            label_boxes.append(pseudo_box[i])
                            label_conf.append(pseudo_conf[i])
                        
                        sorted_indices = np.argsort(label_conf)[::-1]
                        sorted_indices = sorted_indices[:2] if len(sorted_indices) >= 2 else sorted_indices
                        for i in sorted_indices:
                            pseudo_box_.append(label_boxes[i])
                            pseudo_label_ = np.append(pseudo_label_, label)
                        #pseudo_box_.extend(label_boxes[sorted_indices])
                        #pseudo_label_.extend([label] * len(sorted_indices))
                        #pseudo_conf_.extend(label_conf[sorted_indices])
                        
                pseudo_boxes.append(np.array(pseudo_box_))
                pseudo_labels.append(pseudo_label_)
                
                
                #for i in sorted_indices:
                 #   pseudo_box_.append(pseudo_box[i])
                  #  pseudo_label_ = np.append(pseudo_label_, pseudo_label[i])
                        
                #pseudo_boxes.append(np.array(pseudo_box_))
                #pseudo_labels.append(pseudo_label_)
                if len(pseudo_label) == 0:
                    flag = False
                #print(f"iteration: {iteration}    pseudo_labels_num: {len(pseudo_label_)}")

            if not flag:
                continue

            # 使用实例级别标签进行训练
            if len(pseudo_labels) == 0:
                continue
            
            with torch.no_grad():
                if Cuda:
                    images = images.half().cuda()
            
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, pseudo_boxes, pseudo_labels, 1, fp16=True, scaler=scaler)
            total_loss += total.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()

            if iteration % 1000 == 0:
                print(f"pseudo_labels_num: {len(pseudo_labels)}")
                print(pseudo_boxes)
                print(pseudo_labels)
                print(f"epoch: {epoch}    iteration: {iteration}    loss: {total_loss / (iteration + 1)}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        print("model saved!")
        predict()

 