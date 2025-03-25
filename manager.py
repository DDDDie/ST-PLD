import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time 
import random
import pandas as pd
from collections import deque, OrderedDict
from scipy.stats import entropy as kl_divergence

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from utils.functions import save_model, restore_model
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizer
from torch import nn

from utils.metrics import clustering_score
from utils.functions import view_generator
from pretrain import PretrainSTPLDManager
from losses.SupConLoss import SupConLoss
from utils.functions import set_seed

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class STPLDManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        pretrain_manager = PretrainSTPLDManager(args, data, model)  
        
        set_seed(args.seed)
        self.logger = logging.getLogger(logger_name)
        
        loader = data.dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.eval_outputs['loader'], loader.test_outputs['loader']
        self.train_input_ids, self.train_input_mask, self.train_segment_ids = \
            loader.train_outputs['input_ids'], loader.train_outputs['input_mask'], loader.train_outputs['segment_ids']
        self.train_outputs = loader.train_outputs
        self.train_labeled_outputs = loader.train_labeled_outputs
        self.train_labeled_dataloader = loader.train_labeled_outputs['loader']
        
        self.train_examples = loader.train_examples
        self.known_label_list = data.known_label_list
        
        self.criterion = nn.CrossEntropyLoss()
        self.contrast_criterion = SupConLoss()
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)
        
        self.n_known_cls = data.n_known_cls
        
        self.momentum = 0.99
        self.global_threshold_ema = None
        self.class_thresholds_ema = [None] * args.num_labels
        
        self.threshold_history = []
        self.select_num_history = []

        if args.pretrain:
            self.pretrained_model = pretrain_manager.model
            
            self.set_model_optimizer(args, data, model, pretrain_manager)
            self.load_pretrained_model(args, self.pretrained_model)
            
        else:
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))   
            self.set_model_optimizer(args, data, model, pretrain_manager)
            
            if args.train:
                self.load_pretrained_model(args, self.pretrained_model)
            else:
                self.model = restore_model(self.model, args.model_output_dir)   

    def set_model_optimizer(self, args, data, model, pretrain_manager):
        
        if args.cluster_num_factor > 1:
            args.num_labels = self.num_labels = pretrain_manager.num_labels
        else:
            args.num_labels = self.num_labels = data.num_labels
            
        self.model = model.set_model(args, data, 'bert', args.freeze_train_bert_parameters)   
        self.optimizer , self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_examples), args.train_batch_size, \
            args.num_train_epochs, args.lr, args.warmup_proportion)
        self.l_optimizer , self.l_scheduler = model.set_optimizer(self.model, len(data.dataloader.train_labeled_examples), args.train_batch_size, \
            args.num_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device
    
    
    def filter_samples(self, args, feats, km_centroids, assign_labels, pseudo_labels, num_classes, cluster_history, probability_history, last_preds2, delta_label, epoch, labeled_pos, labeled_labels):
        # 根据伪标签的置信度分类别选择给定比例的数据，保证数据集的标签多样性
        select_indices = [] # 存储最终选择的样本索引
        class_scores = [[] for _ in range(num_classes)] # 存储每个类别的样本分数
        class_indices = [[] for _ in range(num_classes)] # 存储每个类别的样本索引
        for i in range(pseudo_labels.size):
            class_indices[pseudo_labels[i]].append(i)
        
        # 指标的分数计算：需要确保分数越大置信度越高
        # distances, 计算每个样本到其聚类中心的距离
        distances = np.sqrt(np.sum((feats - km_centroids[assign_labels]) ** 2, axis=1))
        distances1 = 1 / distances
        

        # 归一化置信度，避免除以零
        max_distances, min_distances = np.max(distances1), np.min(distances1)
        probability = [(distances1[i]-min_distances) / (max_distances-min_distances) for i in range(len(distances1))]
        
        # 每个样本在多次聚类迭代中的概率历史记录
        for i, proba in enumerate(probability):
            # 将当前标签添加到对应样本的聚类历史中
            probability_history[i].append(proba)
            # 如果聚类历史超过2个元素，则移除最旧的元素
            if len(probability_history[i]) > 2:
                probability_history[i].pop(0)
        if epoch < 2:
            label_distribution0 = {}
            select_indices = np.arange(pseudo_labels.size)
            return select_indices, pseudo_labels, label_distribution0
   
        # consistency, 聚类一致性
        cluster_counts = [np.bincount(cluster_history[i], minlength=num_classes) for i in range(len(cluster_history))]
        consistency = [np.max(cluster_counts[i]) / np.sum(cluster_counts[i]) for i in range(len(cluster_history))] # 被分配到次数最多的簇的次数 除以 该样本被分配到所有簇次数的总和
        max_consistency, min_consistency = np.max(consistency), np.min(consistency)
        consistency = [(consistency[i]-min_consistency) / (max_consistency-min_consistency) for i in range(len(consistency))]
                
       
        # 获取已标记样本的伪标签结果
        labeled_pseudo_labels = pseudo_labels[labeled_pos]
        # 建立真实标签和伪标签的映射关系
        label_mapping = {}
        for true_label, pseudo_label in zip(labeled_labels, labeled_pseudo_labels):
            if true_label not in label_mapping:
                label_mapping[true_label] = [pseudo_label]
            else:
                label_mapping[true_label].append(pseudo_label)

        # 对 key 进行排序
        sorted_keys = sorted(label_mapping.keys())
        # 创建一个新的 OrderedDict 来存储排序后的映射关系
        sorted_label_mapping = OrderedDict((key, label_mapping[key]) for key in sorted_keys)
        
        # 原已知意图索引和现聚类后的已知意图索引
        known_intent = np.unique(labeled_labels)
        pseudo_known_intent = []
        
        # 统计每个真实标签对应的伪标签分布
        label_distribution = {}
        for true_label, pseudoo_labels in sorted_label_mapping.items():
            label_distribution[true_label] = {}
            for pseudo_label in pseudoo_labels:
                if pseudo_label not in label_distribution[true_label]:
                    label_distribution[true_label][pseudo_label] = 1
                else:
                    label_distribution[true_label][pseudo_label] += 1
                    
                    
        # 统计每个聚类中标注数据对应的真实标签分布
        label_mapping2 = {}
        for true_label, pseudo_label in zip(labeled_labels, labeled_pseudo_labels):
            if pseudo_label not in label_mapping2:
                label_mapping2[pseudo_label] = [true_label]
            else:
                label_mapping2[pseudo_label].append(true_label)   
        label_distribution2 = {}
        for pseudo_label, true_labels in label_mapping2.items():
            label_distribution2[pseudo_label] = {}
            for true_label in true_labels:
                if true_label not in label_distribution2[pseudo_label]:
                    label_distribution2[pseudo_label][true_label] = 1
                else:
                    label_distribution2[pseudo_label][true_label] += 1

        for i in known_intent:
            # 判断该标注数据是否只分到了一个类别中
            if len(label_distribution[i]) == 1:
                # 记录该类别的现在的伪标签，将后续所有可靠的类别伪标签记录下来
                pseudo_known_intent.append(list(label_distribution[i].keys())[0])
                continue
            else:
                # 如果分到了多个类别中，根据标签分布中的数量判断哪个是已知意图类别
                max_pseudo_label = max(label_distribution[i], key=label_distribution[i].get) 
                pseudo_known_intent.append(max_pseudo_label)
        
        pseudo_known_intent = sorted(np.unique(pseudo_known_intent))
        
        # 初始化聚合程度数组
        cohesion = np.zeros(num_classes)
        # 对于已知意图类别，计算其中标注数据的信息熵和利用上一轮无标注样本弱监督标签的信息熵
        for c in pseudo_known_intent:
            # 该类别的伪标签分布
            pseudo_label_distribution = label_distribution2[c]
            # 该类别的标注数据数量
            num_samples = len([j for j in labeled_pos if assign_labels[j] == c])
            # 该类别的信息熵
            entropy = 0
            for pseudo_label, count in pseudo_label_distribution.items():
                entropy += -count / num_samples * np.log2(count / num_samples)
            # 该类别的聚合程度
            if np.log(len(pseudo_label_distribution)) == 0:
                cohesion[c] = 1
            else:
                cohesion[c] = 1 - entropy / np.log(len(pseudo_label_distribution))      
        
        # 未知意图的类别索引是原来所有类别去掉已知意图索引
        pseudo_new_intent = np.setdiff1d(np.arange(num_classes), pseudo_known_intent)

        # 对于未知意图类别，使用上一轮的伪标签聚类结果作为本轮的弱监督信号来计算其信息熵
        
        # 统计每个样本在本轮的伪标签分布
        label_counts = {}
        current_pseudo_labels = [history[-2] for history in cluster_history]
        # 参考分布：上一迭代的所有样本概率
        pre_probability = [probability_history[-2] for probability_history in probability_history]
    
        stability = np.zeros(num_classes)
        c = 0
        for indices in class_indices:
            labels_in_class = [(current_pseudo_labels[i],i) for i in indices]
            # 从current_probability和probability获取该类别的概率分布
            
            reference_distribution = np.array([pre_probability[idx] for _, idx in labels_in_class])
            current_distribution = np.array([probability[idx] for _, idx in labels_in_class])       
            
            epsilon = 1e-9  # 平滑项
            # 裁剪并替换原始概率值
            current_distribution = np.clip(current_distribution, epsilon, 1 - epsilon)
            reference_distribution = np.clip(reference_distribution, epsilon, 1 - epsilon)

            label_counts = {}
            total_samples = 0
            
            # 统计每个伪标签的加权计数
            for label, indice in labels_in_class:
                if label in label_counts:
                    label_counts[label] += probability[indice]
                else:
                    label_counts[label] = probability[indice]
                total_samples += probability[indice]
                
            # 计算信息熵
            entropy = 0
            for count in label_counts.values():
                probab = count / total_samples
                if probab > 0:  # 确保不会对零取对
                    entropy -= probab * np.log2(probab)
            # 计算稳定性
            num_labels = len(label_counts)
            if num_labels > 0:
                log_num_labels = np.log(num_labels)
                if log_num_labels > 0:
                    stability[c] = 1 - entropy / log_num_labels
                else:
                    stability[c] = 1  
            else:
                stability[c] = 1  

            c += 1

        class_quality = np.zeros(num_classes)
        beta = args.beta
        for c in range(num_classes):

            if c in pseudo_new_intent:
                class_quality[c] = stability[c]
            else:
                class_quality[c] = beta*stability[c] + (1-beta)*cohesion[c]
        
        
        # 计算加权综合得分
        alpha = args.alpha
        sample_quality = [alpha*probability[i] + (1-alpha)*consistency[i] for i in range(len(cluster_history))]

        # 总体的样本置信度等于聚类质量乘以样本质量
        confidences = [class_quality[assign_labels[i]] * sample_quality[i] for i in range(len(cluster_history))]
        
        
        # print(max_distance_list)
        avg_confidences = np.mean(confidences)
        print("avg_confidences:", avg_confidences)

        # 将每个类别的置信度分数存储到对应的类别中
        for i in range(assign_labels.size):
            class_scores[assign_labels[i]].append(confidences[i])
      
        # 初始化或更新全局阈值的EMA（如果是第一次筛选，使用置信度的均值）
        if epoch == 2:
            self.global_threshold_ema = np.mean(confidences)
            self.class_thresholds_ema = [np.mean(class_scores[c]) for c in range(num_classes)]
        else:
            self.global_threshold_ema = self.momentum * self.global_threshold_ema + (1 - self.momentum) * np.mean(confidences)
            # 更新每个类别的局部阈值
            for c in range(num_classes):
                class_confidence = np.mean(class_scores[c])
                self.class_thresholds_ema[c] = self.momentum * self.class_thresholds_ema[c] + (1 - self.momentum) * class_confidence    
        
        self.threshold_history.append(self.global_threshold_ema)
        
        print('Global Threshold:', self.global_threshold_ema)

        # 计算每个类别的综合阈值
        class_composite_thresholds = []
        for c in range(num_classes):
            # 局部阈值EMA的最大归一化
            local_threshold_ema_normalized = self.class_thresholds_ema[c] / max(self.class_thresholds_ema)
            # 局部阈值乘以全局阈值EMA
            class_composite_thresholds.append(local_threshold_ema_normalized * self.global_threshold_ema)
        # print('Class Composite Thresholds:', class_composite_thresholds)
         
        select_class_num = [] # 存储每个类别筛选出来的样本数    
        for c in range(num_classes):
            class_select_indices = []
            # 计算置信度的中位数
            threshold = class_composite_thresholds[c]
            for i in range(len(class_scores[c])):
                if class_scores[c][i] > threshold:
                    class_select_indices.append(class_indices[c][i])
                else:
                    continue
                
            # 如果该簇的类别质量分数较低，那么在此基础上筛选的样本再选取top-k个（差的类别里面选择样本分数较高的）
            # if class_quality[c] < np.mean(class_quality):
            if class_quality[c] < np.quantile(class_quality, 0.7):
                # 在该簇的样本中选择分数较高的top-k个样本, 其中k设置需要根据当前类别的质量分数来调整，差的类别选择的样本数较少
                k = int(len(class_select_indices) * class_quality[c] * 0.5)
                sorted_indices = sorted(class_select_indices, key=lambda i: class_scores[c][class_indices[c].index(i)], reverse=True)
                class_select_indices = sorted_indices[:k]  # 获取top-k个样本的索引
 

            select_class_num.append(len(class_select_indices)) # 统计每个类别筛选出来的样本数
            select_indices.extend(class_select_indices)
        
        # 获取筛选出的噪声样本及其被分配到的类别信息到一个新表格中,包括（样本索引，样本文本内容，样本分配的类别，类别的名称）
      
        self.select_num_history.append(len(select_indices))
        
        # 计算标签变化的比例
        current_preds2 = pseudo_labels
        delta_label2 = np.sum(current_preds2 != last_preds2).astype(np.float32) / current_preds2.shape[0] 
        last_preds2 = np.copy(current_preds2)
        
        k_ratio = args.bottom_k_ratio
        k_num = args.bottom_k_num
        if delta_label>0.005:
            # 找到没有被选择的样本索引
            unselected_indices = [i for i in range(assign_labels.size) if i not in select_indices] 
            # 记录需要进行重分配的n个样本的索引，bottom-k
            # k = int(len(unselected_indices) * k_ratio)
            k = k_num
            # 从未被选择的样本中选择分数最低的K个样本
            sorted_unselected_indices = sorted(unselected_indices, key=lambda i: confidences[i])
            redistributed_indices = sorted_unselected_indices[:k]
            
            # 根据 n 个最近邻样本的类别计算类别概率来纠正样本类别
            n = 20  # 设置 n 值
            nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(feats)
            _, indices = nbrs.kneighbors(feats)
            
            corrected_labels = np.copy(pseudo_labels)
            for i in redistributed_indices:
                neighbor_labels = pseudo_labels[indices[i]]
                label_counts = np.bincount(neighbor_labels, minlength=num_classes)
                original_label = corrected_labels[i]
                max_index = np.argmax(label_counts)
                # 找到所有最大值索引
                max_indices = np.where(label_counts == label_counts[max_index])[0]
                if max_index == original_label:
                    # 找到次大值索引
                    sorted_counts_indices = np.argsort(label_counts)[::-1]
                    next_max_indices = []
                    # 收集所有次大值索引
                    for index in sorted_counts_indices[1:]:
                        if label_counts[index] > 0:
                            next_max_indices.append(index)
                    
                    # 如果有次大值索引，优先选择在 cluster_history[i] 中出现过的
                    if next_max_indices:
                        for next_index in next_max_indices:
                            if next_index in cluster_history[i]:
                                corrected_labels[i] = next_index
                                break
                        else:
                            # 如果没有在 cluster_history[i] 中找到，则选择第一个次大值索引
                            corrected_labels[i] = next_max_indices[0]
                    else:
                        # 如果没有次大值索引，可以选择一个默认操作，例如保持原始标签或选择一个特殊标签
                        corrected_labels[i] = original_label  # 或者选择一个特殊的“未知”标签
                else:
                    # 优先选择在 cluster_history[i] 中出现过的最大值索引
                    for max_idx in max_indices:
                        if max_idx in cluster_history[i]:
                            corrected_labels[i] = max_idx
                            break
                    else:
                        # 如果没有在 cluster_history[i] 中找到，则选择第一个最大值索引
                        corrected_labels[i] = max_index

            pseudo_labels = corrected_labels
            
            select_indices.extend(redistributed_indices) 
            self.logger.info('Pseudo_labels Updated! Redistributed_Samples Num: %d', len(redistributed_indices))
            self.logger.info('delta_label: %f', delta_label2)
                    
        return select_indices, pseudo_labels, label_distribution2
        
    def clustering(self, args, epoch, cluster_history, probability_history, last_preds2, delta_label, init = 'k-means++'):
        
        outputs = self.get_outputs(args, mode = 'train', model = self.model) 
        feats = outputs['feats'] 
        y_true = outputs['y_true'] 
        
        labeled_pos = list(np.where(y_true != -1)[0]) # 找出所有非负标签（即已标记的样本）的索引
        labeled_feats = feats[labeled_pos] # 根据索引提取已标记样本的特征
        labeled_labels = y_true[labeled_pos] # 提取已标记样本的真实标签    
        
        
        # 计算已标记样本的质心,即每个标签的样本特征的平均值     
        labeled_centers = []
        for idx, label in enumerate(np.unique(labeled_labels)):
            label_feats = labeled_feats[labeled_labels == label]
            labeled_centers.append(np.mean(label_feats, axis = 0))
        
        if init == 'k-means++':
            
            self.logger.info('Initializing centroids with K-means++...')
            start = time.time()
            
            km = KMeans(n_clusters = self.num_labels, random_state=args.seed, init = 'k-means++').fit(feats) 
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            end = time.time()
            self.logger.info('K-means++ used %s s', round(end - start, 2))   
                       
        elif init == 'centers':
            
            start = time.time()
            self.centroids 
            km = KMeans(n_clusters = self.num_labels, random_state=args.seed, init = self.centroids.cpu().numpy()).fit(feats) 
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            
            end = time.time()
            self.logger.info('K-means used %s s', round(end - start, 2))
                 
        self.centroids = torch.tensor(km_centroids).to(self.device)
             
        # 修改 pseudo_labels = assign_labels.astype(np.long)
        pseudo_labels = assign_labels.astype(np.int64) # 将分配的标签转为长整型，作为伪标签
        
        # 获取已标记样本的伪标签结果
        labeled_pseudo_labels = pseudo_labels[labeled_pos]
        
        # 每个样本在多次聚类迭代中被分配到的簇的历史记录
        for i, label in enumerate(pseudo_labels):
            # 将当前标签添加到对应样本的聚类历史中
            cluster_history[i].append(label)
            # 如果聚类历史超过5个元素，则移除最旧的元素
            if len(cluster_history[i]) > 5:
                cluster_history[i].pop(0)
        
        # 根据伪标签的置信度选择样本
        valid_indices, pseudo_labels, mapping = self.filter_samples(args, feats, km_centroids, assign_labels, pseudo_labels, self.num_labels, cluster_history, probability_history, last_preds2, delta_label, epoch, labeled_pos, labeled_labels)
            
        return outputs, km_centroids, y_true, assign_labels, pseudo_labels, valid_indices

       
    def train(self, args, data): 
        # 初始化聚类中心和上一次迭代的预测标签
        self.centroids = None
        last_preds = None 
        last_preds2 = None
        delta_label = 1.0
        
        # 时间窗口内所有伪标签的记录
        cluster_history = [[] for _ in range(data.dataloader.num_train_examples)]
        # 保存最近几轮迭代的probability
        probability_history = [[] for _ in range(data.dataloader.num_train_examples)]

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"): 
            
            self.model.train()
            
            for batch in tqdm(self.train_labeled_dataloader, desc="Training(All)"):
                            
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                                    
                # 启用梯度计算                            
                with torch.set_grad_enabled(True):
                    
                    aug_mlp_outputs_a, aug_logits_a = self.model(input_ids, segment_ids, input_mask)
                    aug_mlp_outputs_b, aug_logits_b = self.model(input_ids, segment_ids, input_mask)
                    
                    norm_logits = F.normalize(aug_mlp_outputs_a)
                    norm_aug_logits = F.normalize(aug_mlp_outputs_b)
                    # 将两次的输出连接在一起
                    contrastive_feats = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    loss_contrast = self.contrast_criterion(contrastive_feats, labels = label_ids, temperature = args.train_temperature, device = self.device)
                    # 计算对比损失：用于训练模型使得相似的样本接近，不相似的样本远离
                    loss = loss_contrast
                    # 清零梯度并进行反向传播
                    self.l_optimizer.zero_grad() 
                    loss.backward()
                    # 更新模型参数并调整学习率
                    self.l_optimizer.step()
                    self.l_scheduler.step()  
                            
            init_mechanism = 'k-means++' if epoch == 0 else 'centers'
            # 进行聚类，得到聚类结果和伪标签
            outputs, km_centroids, y_true, assign_labels, pseudo_labels, valid_indices = self.clustering(args, epoch, cluster_history, probability_history, last_preds2, delta_label, init = init_mechanism)
                    
            # 计算标签变化的比例
            current_preds = pseudo_labels
            delta_label = np.sum(current_preds != last_preds).astype(np.float32) / current_preds.shape[0] 
            last_preds = np.copy(current_preds)
            
            if epoch > 0:

                self.logger.info("***** Epoch: %s *****", str(epoch))
                self.logger.info('Training Loss: %f', np.round(tr_loss, 5))
                self.logger.info('Delta Label: %f', delta_label)
                if delta_label < args.tol: # 如果标签变化的比例小于阈值，停止训练
                    self.logger.info('delta_label %s < %f', delta_label, args.tol)  
                    self.logger.info('Reached tolerance threshold. Stop training.')
                    break   
                                
            
            if epoch % 5 == 0:
                pseudo_train_dataloader = self.get_augment_dataloader_ori(args, self.train_outputs, pseudo_labels)
            else:
                pseudo_train_dataloader = self.get_augment_dataloader(args, self.train_outputs, valid_indices, pseudo_labels)
                
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()
            
            for batch in tqdm(pseudo_train_dataloader, desc="Training(All)"):
                # 将输入数据移到设备GPU上
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                                        
                with torch.set_grad_enabled(True):
                    
                    input_ids_a,  input_ids_b = self.batch_chunk(input_ids)
                    input_mask_a,  input_mask_b = self.batch_chunk(input_mask)
                    segment_ids_a,  segment_ids_b = self.batch_chunk(segment_ids)
                    label_ids = torch.chunk(input=label_ids, chunks=2, dim=1)[0][:, 0]
                        
                    aug_mlp_outputs_a, aug_logits_a = self.model(input_ids_a, segment_ids_a, input_mask_a)               
                    aug_mlp_outputs_b, aug_logits_b = self.model(input_ids_b, segment_ids_b, input_mask_b)
                    
                    norm_logits = F.normalize(aug_mlp_outputs_a)
                    norm_aug_logits = F.normalize(aug_mlp_outputs_b)
                    # 交叉熵损失
                    loss_ce = 0.5 * (self.criterion(aug_logits_a, label_ids) + self.criterion(aug_logits_b, label_ids)) 
                    # 对比损失
                    contrastive_feats = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    loss_contrast = self.contrast_criterion(contrastive_feats, labels = label_ids, temperature = args.train_temperature, device = self.device)
                    
                    loss = loss_contrast + loss_ce

                    self.optimizer.zero_grad()
                    loss.backward()

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
            
            tr_loss = tr_loss / nb_tr_steps
        
        # 保存模型
        if args.save_model:
            save_model(self.model, args.model_output_dir)
              
    def test(self, args, data):
        # 获取测试数据的特征（feats）和真实标签（y_true）
        test_results = {}
        outputs = self.get_outputs(args, mode = 'test', model = self.model)
        feats = outputs['feats']
        y_true = outputs['y_true']
        
        if args.cluster_num_factor > 1: 
            test_results['estimate_k'] = args.num_labels
        # 使用KMeans聚类算法对特征进行拟合
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed, init = self.centroids.cpu().numpy()).fit(feats) 
        y_pred = km.labels_ 
        
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred) 
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        self.logger.info("%s",str(self.threshold_history))
        self.logger.info("%s",str(self.select_num_history))
                         
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def get_outputs(self, args, mode, model):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, logits = model(input_ids, segment_ids, input_mask, feature_ext = True)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))
        
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        
        y_logits = total_logits.cpu().numpy()
        
        outputs = {
            'y_true': y_true,
            'y_pred': y_pred,
            'logits': y_logits,
            'feats': feats
        }
        return outputs

    def load_pretrained_model(self, args, pretrained_model):
        
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['mlp_head.bias','mlp_head.0.bias',  'classifier.weight', 'classifier.bias', 'mlp_head.0.weight', 'mlp_head.weight'] 
        
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def batch_chunk(self, x):
        x1, x2 = torch.chunk(input=x, chunks=2, dim=1)
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        return x1, x2
    
    def get_augment_dataloader(self, args, train_outputs, valid_indices, pseudo_labels = None):
        
        input_ids = train_outputs['input_ids'][valid_indices]
        input_mask = train_outputs['input_mask'][valid_indices]
        segment_ids = train_outputs['segment_ids'][valid_indices]
        if pseudo_labels is None:
            pseudo_labels = train_outputs['label_ids'][valid_indices]
        
        
        input_ids_a, input_mask_a = self.generator.random_token_erase(input_ids, input_mask)
        input_ids_b, input_mask_b = self.generator.random_token_erase(input_ids, input_mask)
        
        train_input_ids = torch.cat(([input_ids_a.unsqueeze(1), input_ids_b.unsqueeze(1)]), dim = 1)
        train_input_mask = torch.cat(([input_mask_a.unsqueeze(1), input_mask_a.unsqueeze(1)]), dim = 1)
        train_segment_ids = torch.cat(([segment_ids.unsqueeze(1), segment_ids.unsqueeze(1)]), dim = 1)
        
        train_label_ids = torch.tensor(pseudo_labels[valid_indices]).unsqueeze(1)
        train_label_ids = torch.cat(([train_label_ids, train_label_ids]), dim = 1)

        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)

        sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)

        self.logger.info('Dataloader Updated! Select_sample Num: %d', len(valid_indices))
        
        return train_dataloader
    
    def get_augment_dataloader_ori(self, args, train_outputs, pseudo_labels = None):
        
        input_ids = train_outputs['input_ids']
        input_mask = train_outputs['input_mask']
        segment_ids = train_outputs['segment_ids']
        if pseudo_labels is None:
            pseudo_labels = train_outputs['label_ids']
        
        
        input_ids_a, input_mask_a = self.generator.random_token_erase(input_ids, input_mask)
        input_ids_b, input_mask_b = self.generator.random_token_erase(input_ids, input_mask)
        
        train_input_ids = torch.cat(([input_ids_a.unsqueeze(1), input_ids_b.unsqueeze(1)]), dim = 1)
        train_input_mask = torch.cat(([input_mask_a.unsqueeze(1), input_mask_a.unsqueeze(1)]), dim = 1)
        train_segment_ids = torch.cat(([segment_ids.unsqueeze(1), segment_ids.unsqueeze(1)]), dim = 1)
        
        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
        train_label_ids = torch.cat(([train_label_ids, train_label_ids]), dim = 1)

        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)

        sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)

        return train_dataloader