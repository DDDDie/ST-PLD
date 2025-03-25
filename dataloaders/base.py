import numpy as np
import os   
import logging

from .__init__ import max_seq_lengths, backbone_loader_map, benchmark_labels # 从当前目录下的__init__.py文件中导入三个对象


class DataManager:
    
    def __init__(self, args, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        self.all_label_list = self.get_labels(args.dataset) 

        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio) # 计算已知类别的数量：将所有标签的数量乘以args.known_cls_ratio，然后四舍五入
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False)) # 从所有标签中随机选择self.n_known_cls个标签，形成已知标签列表
        # 记录到日志中
        self.logger.info('The number of known intents is %s', self.n_known_cls)
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list))

        args.num_labels = self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor) # 簇数K的因子(放大倍数)
        
        self.dataloader = self.get_loader(args, self.get_attrs())

                
    def get_labels(self, dataset): # 用于获取指定数据集的标签
        
        labels = benchmark_labels[dataset]

        return labels
    
    def get_loader(self, args, attrs): # 根据特定的参数和属性来定义数据加载器
        
        dataloader = backbone_loader_map[args.backbone](args, attrs)

        return dataloader
    
    def get_attrs(self): # 获取DataManager的所有属性

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs



