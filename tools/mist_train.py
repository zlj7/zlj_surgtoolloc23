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
import matplotlib.pyplot as plt
import wandb

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
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.999), weight_decay = 0)
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

    loss_record = []
    
    #---------------------------------------#
    #   wandb可视化
    #---------------------------------------#
    wandb.init(project="surtool23_self_train")
    wandb.config.batch_size = 1
    wandb.config.lr = 0.0001
    wandb.watch(model, log="all")
        
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
            cnt = 0
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
                # 添加上batch_size维度
                image_data  = np.expand_dims(image, 0)
                image_data = torch.from_numpy(image_data).float()
                if Cuda:
                    image_data = image_data.cuda()

                roi_cls_locs, roi_scores, rois, _ = model(image_data)  # 取出roi预测框和概率
                
                # 利用classifier的预测结果对建议框进行解码，获得预测框
                with torch.no_grad():
                    results = decodebox.forward(roi_cls_locs, roi_scores, rois, (720, 1280), image_shape, nms_iou = 0.5, confidence = 0.5)

                if len(results[0]) > 0:
                    # top_label = np.array(results[0][:, 5], dtype='int32') #[0, 3]的类型，代表预测框的编号
                    top_label = results[0][:, 5]
                    top_conf = results[0][:, 4] #[0.97, 0.82] 代表概率
                    top_boxes = results[0][:, :4] #top, left, bottom, right
                    
                    # 改成left, top, right, bottom
                    left, top, right, bottom = top_boxes[:, [1, 0, 3, 2]].T
                    top_boxes = np.array([left, top, right, bottom]).T
                    
                    # 过滤掉两边的黑边
                    #top_boxes_ = top_boxes[np.logical_and(top_boxes[:,0]>130, top_boxes[:,2]<1050)]
                    #top_conf_ = top_conf[np.logical_and(top_boxes[:,0]>130, top_boxes[:,2]<1050)]
                    #top_label_ = top_label[np.logical_and(top_boxes[:,0]>130, top_boxes[:,2]<1050)]
                    # 过滤掉上下的黑边
                    #top_boxes_1 = top_boxes_[np.logical_and(top_boxes_[:,1]>50, top_boxes_[:,3]<680)]
                    #top_conf_1 = top_conf_[np.logical_and(top_boxes_[:,1]>50, top_boxes_[:,3]<680)]
                    #top_label_1 = top_label_[np.logical_and(top_boxes_[:,1]>50, top_boxes_[:,3]<680)]
                    
                    # 过滤掉太小的box
                    box_w = top_boxes[:, 2] - top_boxes[:, 0]
                    box_h = top_boxes[:, 3] - top_boxes[:, 1]
                    top_boxes = top_boxes[np.logical_and(box_w>100, box_h>100)]
                    top_conf = top_conf[np.logical_and(box_w>100, box_h>100)]
                    top_label = top_label[np.logical_and(box_w>100, box_h>100)]
                    
                else:
                    flag = False
                    top_label = []
                    top_conf = []
                    top_boxes = []
                    #continue
                
                # 取出在label中标注的类
                for i in range(len(top_label)):
                    if top_label[i] in labels[cnt]:
                        pseudo_box.append(top_boxes[i])
                        pseudo_label = np.append(pseudo_label, top_label[i])
                        pseudo_conf = np.append(pseudo_conf, top_conf[i])
                
                cnt += 1        
                
                pseudo_box_ = []
                pseudo_label_ = []
                # 取出label中每类概率最高的2个
                #unique_labels = np.unique(pseudo_label)
                
                #for label in unique_labels:
                 #   label_indices = np.where(pseudo_label == label)[0]
                  #  if len(label_indices) > 0:
                   #     label_boxes = []
                    #    label_conf = []
                     #   for i in label_indices:
                      #      label_boxes.append(pseudo_box[i])
                       #     label_conf.append(pseudo_conf[i])
                        
                        #sorted_indices = np.argsort(label_conf)[::-1]
                        #sorted_indices = sorted_indices[:2] if len(sorted_indices) >= 2 else sorted_indices
                        #for i in sorted_indices:
                         #   pseudo_box_.append(label_boxes[i])
                          #  pseudo_label_ = np.append(pseudo_label_, label)
                              
                # 取出概率最高的3个
                #sorted_indices = np.argsort(pseudo_conf)[::-1] # 按概率排序
                #sorted_indices = sorted_indices[:3] # 取概率最高的3个，不满3个取全部    
                #for i in sorted_indices:
                 #   pseudo_box_.append(pseudo_box[i])
                  #  pseudo_label_ = np.append(pseudo_label_, pseudo_label[i])
                  
                # 根据labels中的类别数量取box(如[0,0,1]取两个0,一个1)
                unique_labels, class_counts = np.unique(labels.cpu(), return_counts=True)

                for label, count in zip(unique_labels, class_counts):
                    label_indices = np.where(pseudo_label == label)[0]
                    if len(label_indices) > 0:
                        label_boxes = []
                        label_conf = []
                        for i in label_indices:
                            label_boxes.append(pseudo_box[i])
                            label_conf.append(pseudo_conf[i])
                
                        sorted_indices = np.argsort(label_conf)[::-1]
                        max_boxes = min(count, len(sorted_indices))
                        sorted_indices = sorted_indices[:max_boxes] if max_boxes >= 1 else sorted_indices
                        for i in sorted_indices:
                            pseudo_box_.append(label_boxes[i])
                            pseudo_label_ = np.append(pseudo_label_, label)
                        
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

            if iteration % 10 == 0:
                print(f"epoch: {epoch}    iteration: {iteration}    loss: {total_loss / (iteration + 1)}")
                
            if iteration % 100 == 0:
                print(pseudo_boxes)
                print(pseudo_labels)
                print(f"epoch: {epoch}    iteration: {iteration}    loss: {total_loss / (iteration + 1)}")

                torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
                print("model saved!")
                predict()
                
        print(f"epoch {epoch} finished!")
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        print("model saved!")
        predict()
        
        loss_record.append(total_loss / (iteration + 1))
        wandb.log({'loss': total_loss / (iteration + 1)})
        wandb.save('model.h5')
        
        # 绘制图像
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(np.arange(len(loss_record)), loss_record,label='total_loss')
        axes.set_xlabel('epochs')
        axes.set_ylabel('loss')
        axes.set_title('Loss')
        axes.legend()
        f = plt.gcf()  #获取当前图像
        f.savefig('self_train_loss.png')
        f.clear()  #释放内存

 