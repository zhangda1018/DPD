import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.locator import Crowd_locator
from model.PBM import BinarizedModule
from config import cfg
from misc.utils import *
import datasets
import cv2, sys
from tqdm import tqdm
from misc.compute_metric import eval_metrics
from misc.EMA import EMA

class Trainer():
    def __init__(self, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.src_train_loader, self.src_val_loader, self.src_restore_transform = datasets.loading_data('QNRF')
        self.tra_train_loader, self.val_loader, self.tra_restore_transform = datasets.loading_data('FDST')
        # _, self.JHU_loader, _ = datasets.loading_data('JHU')
        # _, self.NWPU_loader, _ = datasets.loading_data('NWPU')
        # _, self.QNRF_loader, _ = datasets.loading_data('QNRF')
        # _, self.FDST_loader, _ = datasets.loading_data('FDST')
        _, self.SHHB_loader, _ = datasets.loading_data('FDST')


        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = Crowd_locator(cfg.NET,cfg.GPU_ID,pretrained=True)
        self.pseudo_head = BinarizedModule(768).cuda()
        self.worst_head = BinarizedModule(768).cuda()
        state_dict = torch.load('/data/gjy/Projects/DPD/models/QNRF_pretrained.pth')
        # state_dict = torch.load('/data2/gjy_data/IIM/Projects/exp/09-13_11-34_SHHA_VGG16_FPN/ep_157_F1_0.781_Pre_0.804_Rec_0.759_mae_18.6_mse_35.4.pth')
        # print(list(self.net.state_dict().keys())[10:20], list(state_dict.keys())[10:20])
        if len(cfg.GPU_ID) > 1:
            try:
                self.net.load_state_dict(state_dict)
            except:
                new_dict = {}
                for key in state_dict:
                    new_dict[key.replace(key.split('.')[0], key.split('.')[0]+'.module')] = state_dict[key]
                self.net.load_state_dict(new_dict)
        else:
            try:
                self.net.load_state_dict(state_dict)
            except:
                new_dict = {}
                for key in state_dict:
                    new_dict[key.replace('.module', '')] = state_dict[key]
                self.net.load_state_dict(new_dict)
        self.ema = EMA(self.net, 0.99)
        self.ema.register()
        if cfg.OPT == 'Adam':
            self.optimizer = optim.Adam([{'params':self.net.Extractor.parameters(), 'lr':cfg.LR_BASE_NET, 'weight_decay':1e-5},
                                         {'params':self.net.Binar.parameters(), 'lr':cfg.LR_BM_NET},
                                        #  {'params':self.pseudo_head.parameters(), 'lr':cfg.LR_BM_NET},
                                         {'params':self.worst_head.parameters(), 'lr':cfg.LR_BM_NET}])

        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.train_record = {'best_F1': 0, 'best_Pre': 0,'best_Rec': 0, 'best_mae': 1e20, 'best_mse':1e20, 'best_nae':1e20, 'best_model_name': ''}
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0
        self.num_iters = cfg.MAX_EPOCH * int(len(self.src_train_loader))
        with open('Class Recall.txt', 'w') as f:
            f.write("Class Recall Record\n")
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.num_iters = latest_state['num_iters']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
            print("Finish loading resume mode")
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ['exp','figure','img', 'vis'], resume=cfg.RESUME)

    def record_init(self):
        for name in ['SHHA', 'SHHB', 'QNRF', 'JHU']:
            with open(name+'_Test.txt', 'w') as f:
                f.write('Ep  F1-m  Prec.    Recal   MAE    rMSE     NAE\n')

    def record_information(self, ep, name, infos):
        ep = str(ep)
        content = ep + ' '
        for elem in range(len(infos)):
            content += str(infos[elem])
            content += ' '
        content += '\n'
        with open(f'{name}_Test.txt', 'a') as f:
            f.write(content)

    def forward(self):
        self.record_init()
        # SHHA_f1m_l, SHHA_ap_l, SHHA_ar_l, SHHA_mae, SHHA_mse, SHHA_nae = self.validate(self.val_loader)
        # JHU_f1m_l, JHU_ap_l, JHU_ar_l, JHU_mae, JHU_mse, JHU_nae = self.validate(self.JHU_loader)
        # QNRF_f1m_l, QNRF_ap_l, QNRF_ar_l, QNRF_mae, QNRF_mse, QNRF_nae = self.validate(self.QNRF_loader)
        # # FDST_f1m_l, FDST_ap_l, FDST_ar_l, FDST_mae, FDST_mse, FDST_nae = self.validate(self.FDST_loader)
        SHHB_f1m_l, SHHB_ap_l, SHHB_ar_l, SHHB_mae, SHHB_mse, SHHB_nae = self.validate(self.SHHB_loader)
        # for name in ['SHHB']:
        #     self.record_information(self.epoch, name, [eval(f'{name}_f1m_l'), eval(f'{name}_ap_l'), eval(f'{name}_ar_l'),
        #                                             eval(f'{name}_mae'), eval(f'{name}_mse'), eval(f'{name}_nae')])
        
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            
            self.source_loader = enumerate(self.src_train_loader)
            self.epoch = epoch
            # training    
            self.timer['train time'].tic()
            self.MultiHeadDebiased_train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if 1:#epoch%cfg.VAL_FREQ==0 and epoch>cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                # SHHA_f1m_l, SHHA_ap_l, SHHA_ar_l, SHHA_mae, SHHA_mse, SHHA_nae = self.validate(self.val_loader)
                # JHU_f1m_l, JHU_ap_l, JHU_ar_l, JHU_mae, JHU_mse, JHU_nae = self.validate(self.JHU_loader)
                # # NWPU_f1m_l, NWPU_ap_l, NWPU_ar_l, NWPU_mae, NWPU_mse, NWPU_nae = self.validate(self.NWPU_loader)
                # QNRF_f1m_l, QNRF_ap_l, QNRF_ar_l, QNRF_mae, QNRF_mse, QNRF_nae = self.validate(self.QNRF_loader)
                # FDST_f1m_l, FDST_ap_l, FDST_ar_l, FDST_mae, FDST_mse, FDST_nae = self.validate(self.FDST_loader)
                SHHB_f1m_l, SHHB_ap_l, SHHB_ar_l, SHHB_mae, SHHB_mse, SHHB_nae = self.validate(self.SHHB_loader)
                # for name in ['SHHB']:
                #     self.record_information(self.epoch, name, [eval(f'{name}_f1m_l'), eval(f'{name}_ap_l'), eval(f'{name}_ar_l'),
                #                                                 eval(f'{name}_mae'), eval(f'{name}_mse'), eval(f'{name}_nae')])
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )

            # if epoch > cfg.LR_DECAY_START:
            #     self.scheduler.step()


    def train(self): # training for all datasets
        self.net.train()

        for i, data in enumerate(self.src_train_loader, 0):
            tra_data = iter(self.src_train_loader).next()
            self.i_tb+=1
            if self.i_tb % 40 == 0:
                self.validate()
                self.net.train()
            self.timer['iter time'].tic()
            img, strong_img, gt_map = data
            tra_img, tra_strong_img, tra_gt = tra_data
            
            tra_img = Variable(tra_img).cuda()
            tra_strong_img = Variable(tra_strong_img).cuda()
            tra_gt = Variable(tra_gt).cuda()


            img = Variable(img).cuda()
            # strong_img = Variable(strong_img).cuda()
            gt_map = Variable(gt_map).cuda()
            batch_size = img.size(0)

            mix_img, mix_gt = torch.cat((img, tra_img)), torch.cat((gt_map, tra_gt))

            while True:
                T, P, B = self.net(mix_img, mix_gt)
                threshold_matrix, pre_map, binar_map = T[: batch_size], P[: batch_size], B[: batch_size]
                head_map_loss, binar_map_loss = F.mse_loss(pre_map, gt_map), torch.abs(binar_map - gt_map).mean()
                sup_loss = head_map_loss + binar_map_loss
                
                tra_threshold_matrix, tra_pre_map, tra_binar_map = T[batch_size: ], P[batch_size: ], B[batch_size: ]
                tra_head_map_loss, tra_binar_map_loss = F.mse_loss(tra_pre_map, tra_gt), torch.abs(tra_binar_map - tra_gt).mean()
                manul_loss = tra_head_map_loss + tra_binar_map_loss
                break

            while True:
                with torch.no_grad():
                    self.ema.apply_shadow()
                    psu_threshold_matrix, psu_pre_map, psu_binar_map = self.net(tra_img, mask_gt = None, mode='val')
                    self.ema.restore()
                    psu_pre_map, psu_binar_map = psu_pre_map.detach(), psu_binar_map.detach()
                break

            consis_loss = F.mse_loss(tra_pre_map, psu_pre_map) + torch.abs(tra_binar_map - psu_binar_map).mean()

            all_loss = consis_loss + sup_loss
            


            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            lr1,lr2 = adjust_learning_rate(self.optimizer,
                            cfg.LR_BASE_NET,
                            cfg.LR_BM_NET,
                            self.num_iters,
                            self.i_tb)
            sys.stdout.write('\r')
            sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d] Loss: [%.4f, %.4f | %.4f] Thre: [%.3f|%.3f]'
                    %(self.epoch + 1, 600, i + 1, len(self.src_train_loader), consis_loss.item(), sup_loss.item(), manul_loss.item(),
                    tra_threshold_matrix.mean().item(), psu_threshold_matrix.mean().item()))
            sys.stdout.flush()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.writer.add_scalar('train_lr1', lr1, self.i_tb)
                self.writer.add_scalar('train_lr2', lr2, self.i_tb)
                self.writer.add_scalar('train_loss', head_map_loss.item(), self.i_tb)
                self.writer.add_scalar('Binar_loss', binar_map_loss.item(), self.i_tb)
                if len(cfg.GPU_ID)>1:
                    self.writer.add_scalar('weight', self.net.Binar.module.weight.data.item(), self.i_tb)
                    self.writer.add_scalar('bias', self.net.Binar.module.bias.data.item(), self.i_tb)
                else:
                    self.writer.add_scalar('weight', self.net.Binar.weight.data.item(), self.i_tb)
                    self.writer.add_scalar('bias', self.net.Binar.bias.data.item(), self.i_tb)

                self.timer['iter time'].toc(average=False)
            
            if  i %100==0:
                box_pre, boxes = self.get_boxInfo_from_Binar_map(binar_map[0].detach().cpu().numpy())
                vis_results('tmp_vis', 0, self.writer, self.tra_restore_transform, tra_img, tra_pre_map[0].detach().cpu().numpy(), \
                                 tra_gt[0].detach().cpu().numpy(),tra_binar_map.detach().cpu().numpy(),
                                 tra_threshold_matrix.detach().cpu().numpy(),boxes)

    def dice_loss(self, target, predictive, ep=1e-8):
        intersection = 2 * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

    def MultiHeadDebiased_train(self):
        self.net.train()

        for i, data in enumerate(self.src_train_loader, 0):
            tra_data = self.source_loader.__next__()
            self.i_tb+=1
            # if self.i_tb % 40 == 0:
            #     self.validate()
            #     self.net.train()
            self.timer['iter time'].tic()
            img, strong_img, gt_map = data
            _, eff_data = tra_data
            tra_img, tra_strong_img, tra_gt = eff_data
            
            tra_img = Variable(tra_img).cuda()
            tra_strong_img = Variable(tra_strong_img).cuda()
            tra_gt = Variable(tra_gt).cuda()


            img = Variable(img).cuda()
            # strong_img = Variable(strong_img).cuda()
            gt_map = Variable(gt_map).cuda()
            batch_size = img.size(0)

            mix_img, mix_gt = torch.cat((img, tra_img)), torch.cat((gt_map, tra_gt))

            while True:
                T, P, B, Feat = self.net(mix_img, None, 'pseudo')
                threshold_matrix, pre_map, binar_map = T[: batch_size], P[: batch_size], B[: batch_size]
                head_map_loss, binar_map_loss = F.mse_loss(pre_map, gt_map), torch.abs(binar_map - gt_map).mean()
                sup_loss = head_map_loss + binar_map_loss
                
                tra_threshold_matrix, tra_pre_map, tra_binar_map = T[batch_size: ], P[batch_size: ], B[batch_size: ]
                tra_feature = Feat[batch_size: ]
                tra_head_map_loss, tra_binar_map_loss = F.mse_loss(tra_pre_map, tra_gt), torch.abs(tra_binar_map - tra_gt).mean()
                manul_loss = tra_head_map_loss + tra_binar_map_loss
                break

            while True:
                _, psuedo_binar_map = self.pseudo_head(tra_feature, tra_pre_map)
                # DICE_loss = self.dice_loss(tra_binar_map.detach(), psuedo_binar_map)
                # IOU_loss = soft_IOULoss(psuedo_binar_map, tra_binar_map.detach())
                L2_loss = F.mse_loss(psuedo_binar_map, tra_binar_map.detach())
                DEbias_L1_loss = torch.abs(psuedo_binar_map - tra_binar_map.detach()).mean()
                debiased_loss = L2_loss + DEbias_L1_loss#torch.mean(psuedo_binar_map - tra_binar_map.detach()).mean()
                break
                    

            while True:
                with torch.no_grad():
                    self.ema.apply_shadow()
                    psu_threshold_matrix, psu_pre_map, psu_binar_map = self.net(tra_img, mask_gt = None, mode='val')
                    self.ema.restore()
                    psu_pre_map, psu_binar_map = psu_pre_map.detach(), psu_binar_map.detach()
                break
            
            

            consis_loss = F.mse_loss(tra_pre_map, psu_pre_map) + torch.abs(tra_binar_map - psu_binar_map).mean()

            all_loss = consis_loss + sup_loss + debiased_loss
            


            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            lr1,lr2 = adjust_learning_rate(self.optimizer,
                            cfg.LR_BASE_NET,
                            cfg.LR_BM_NET,
                            self.num_iters,
                            self.i_tb)
            sys.stdout.write('\r')
            sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d] Loss: [%.4f, %.4f |, %.4f %.4f] Thre: [%.3f|%.3f]'
                    %(self.epoch + 1, 600, i + 1, len(self.src_train_loader), 
                    consis_loss.item(), sup_loss.item(), L2_loss.item(), DEbias_L1_loss.item(),
                    tra_threshold_matrix.mean().item(), psu_threshold_matrix.mean().item()))
            sys.stdout.flush()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.writer.add_scalar('train_lr1', lr1, self.i_tb)
                self.writer.add_scalar('train_lr2', lr2, self.i_tb)
                self.writer.add_scalar('train_loss', head_map_loss.item(), self.i_tb)
                self.writer.add_scalar('Binar_loss', binar_map_loss.item(), self.i_tb)
                if len(cfg.GPU_ID)>1:
                    self.writer.add_scalar('weight', self.net.Binar.module.weight.data.item(), self.i_tb)
                    self.writer.add_scalar('bias', self.net.Binar.module.bias.data.item(), self.i_tb)
                else:
                    self.writer.add_scalar('weight', self.net.Binar.weight.data.item(), self.i_tb)
                    self.writer.add_scalar('bias', self.net.Binar.bias.data.item(), self.i_tb)

                self.timer['iter time'].toc(average=False)
            
            if  i %100==0:
                box_pre, boxes = self.get_boxInfo_from_Binar_map(binar_map[0].detach().cpu().numpy())
                vis_results('tmp_vis', 0, self.writer, self.tra_restore_transform, tra_img, tra_pre_map[0].detach().cpu().numpy(), \
                                 tra_gt[0].detach().cpu().numpy(),tra_binar_map.detach().cpu().numpy(),
                                 tra_threshold_matrix.detach().cpu().numpy(),boxes) 

    def Debiased_train(self):
        self.net.train()

        for i, data in enumerate(self.src_train_loader, 0):
            tra_data = iter(self.src_train_loader).next()
            self.i_tb+=1
            self.timer['iter time'].tic()
            img, strong_img, gt_map = data
            tra_img, tra_strong_img, tra_gt = tra_data
            
            tra_img = Variable(tra_img).cuda()
            tra_strong_img = Variable(tra_strong_img).cuda()
            tra_gt = Variable(tra_gt).cuda()


            img = Variable(img).cuda()
            # strong_img = Variable(strong_img).cuda()
            gt_map = Variable(gt_map).cuda()
            batch_size = img.size(0)

            mix_img, mix_gt = torch.cat((img, tra_img)), torch.cat((gt_map, tra_gt))

            while True:
                T, P, B, Feat = self.net(mix_img, None, 'pseudo')
                threshold_matrix, pre_map, binar_map = T[: batch_size], P[: batch_size], B[: batch_size]
                head_map_loss, binar_map_loss = self.dice_loss(pre_map, gt_map, mode='L2'), self.dice_loss(binar_map, gt_map, mode='L1')
                sup_loss = head_map_loss + binar_map_loss
                
                tra_threshold_matrix, tra_pre_map, tra_binar_map = T[batch_size: ], P[batch_size: ], B[batch_size: ]
                src_feature, tra_feature = Feat[: batch_size], Feat[batch_size: ]
                tra_head_map_loss, tra_binar_map_loss = F.mse_loss(tra_pre_map, tra_gt), torch.abs(tra_binar_map - tra_gt).mean()
                manul_loss = tra_head_map_loss + tra_binar_map_loss
                break

            while True:
                _, psuedo_binar_map = self.pseudo_head(tra_feature, tra_pre_map)
                debiased_loss = self.dice_loss(psuedo_binar_map, tra_binar_map.detach(), mode='L1')
                break
            
            while True:
                for param in self.worst_head.parameters():
                    param.requires_grad = False
                _, tra_worst_binar_map = self.worst_head(tra_feature, tra_pre_map)
                _, src_worst_binar_map = self.worst_head(src_feature, pre_map)
                worst_loss = -self.dice_loss(src_worst_binar_map, binar_map.detach(), mode='L1') + \
                            self.dice_loss(tra_worst_binar_map, tra_binar_map.detach(), mode='L1')
                break
            
            while True:
                for param in self.worst_head.parameters():
                    param.requires_grad = True
                _, tra_worst_binar_map = self.worst_head(tra_feature.detach(), tra_pre_map.detach())
                _, src_worst_binar_map = self.worst_head(src_feature.detach(), pre_map.detach())
                worst_game_loss = self.dice_loss(src_worst_binar_map, binar_map.detach(), mode='L1') - \
                            self.dice_loss(tra_worst_binar_map, tra_binar_map.detach(), mode='L1')
                break
            while True:
                with torch.no_grad():
                    self.ema.apply_shadow()
                    psu_threshold_matrix, psu_pre_map, psu_binar_map = self.net(tra_img, mask_gt = None, mode='val')
                    self.ema.restore()
                    psu_pre_map, psu_binar_map = psu_pre_map.detach(), psu_binar_map.detach()
                break
            
            

            consis_loss = self.dice_loss(tra_pre_map, psu_pre_map, mode='L2') + self.dice_loss(tra_binar_map, psu_binar_map, mode='L1')

            all_loss = consis_loss + sup_loss + 2 * debiased_loss + worst_game_loss + worst_loss
            


            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            lr1,lr2 = adjust_learning_rate(self.optimizer,
                            cfg.LR_BASE_NET,
                            cfg.LR_BM_NET,
                            self.num_iters,
                            self.i_tb)
            sys.stdout.write('\r')
            sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d] Loss: [%.4f, %.4f, %.4f | %.4f, %.4f] Thre: [%.3f|%.3f]'
                    %(self.epoch + 1, 600, i + 1, len(self.src_train_loader), 
                    consis_loss.item(), sup_loss.item(), debiased_loss.item(), worst_game_loss.item(), worst_loss.item(),
                    tra_threshold_matrix.mean().item(), psu_threshold_matrix.mean().item()))
            sys.stdout.flush()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.writer.add_scalar('train_lr1', lr1, self.i_tb)
                self.writer.add_scalar('train_lr2', lr2, self.i_tb)
                self.writer.add_scalar('train_loss', head_map_loss.item(), self.i_tb)
                self.writer.add_scalar('Binar_loss', binar_map_loss.item(), self.i_tb)
                if len(cfg.GPU_ID)>1:
                    self.writer.add_scalar('weight', self.net.Binar.module.weight.data.item(), self.i_tb)
                    self.writer.add_scalar('bias', self.net.Binar.module.bias.data.item(), self.i_tb)
                else:
                    self.writer.add_scalar('weight', self.net.Binar.weight.data.item(), self.i_tb)
                    self.writer.add_scalar('bias', self.net.Binar.bias.data.item(), self.i_tb)

                self.timer['iter time'].toc(average=False)
            
            if  i %100==0:
                box_pre, boxes = self.get_boxInfo_from_Binar_map(binar_map[0].detach().cpu().numpy())
                vis_results('tmp_vis', 0, self.writer, self.tra_restore_transform, tra_img, tra_pre_map[0].detach().cpu().numpy(), \
                                 tra_gt[0].detach().cpu().numpy(),tra_binar_map.detach().cpu().numpy(),
                                 tra_threshold_matrix.detach().cpu().numpy(),boxes) 

    def get_boxInfo_from_Binar_map(self, Binar_numpy, min_area=3):
        Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
        assert Binar_numpy.ndim == 2
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

        boxes = stats[1:, :]
        points = centroids[1:, :]
        index = (boxes[:, 4] >= min_area)
        boxes = boxes[index]
        points = points[index]
        pre_data = {'num': len(points), 'points': points}
        return pre_data, boxes

    def validate(self, loader):
        self.net.eval()
        num_classes = 6
        losses = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
        metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                     'fn_c': AverageCategoryMeter(num_classes)}
        metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                     'fn_c': AverageCategoryMeter(num_classes)}

        c_maes = {'level': AverageCategoryMeter(5), 'illum': AverageCategoryMeter(4)}
        c_mses = {'level': AverageCategoryMeter(5), 'illum': AverageCategoryMeter(4)}
        c_naes = {'level': AverageCategoryMeter(5), 'illum': AverageCategoryMeter(4)}
        gen_tqdm = tqdm(loader)
        recall_c_list = [[] for _ in range(num_classes)]
        for vi, data in enumerate(gen_tqdm, 0):
            img,dot_map, gt_data = data
            slice_h, slice_w = 512, 512#self.cfg_data.TRAIN_SIZE

            with torch.no_grad():
                img = Variable(img).cuda()
                dot_map = Variable(dot_map).cuda()
                # crop the img and gt_map with a max stride on x and y axis
                # size: HW: __C_NWPU.TRAIN_SIZE
                # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
                crop_imgs, crop_gt, crop_masks = [], [], []
                b, c, h, w = img.shape

                if h*w< slice_h*2*slice_w*2 and h%16 == 0 and w %16 == 0:
                    [pred_threshold, pred_map, __]= [i.cpu() for i in self.net(img, mask_gt=None, mode = 'val')]
                else:
                    if h % 16 !=0:
                        pad_dims = (0,0, 0,16-h%16)
                        h = (h//16+1)*16
                        img = F.pad(img, pad_dims, "constant")
                        dot_map = F.pad(dot_map, pad_dims, "constant")

                    if w % 16 !=0:
                        pad_dims = (0, 16-w%16, 0, 0)
                        w =  (w//16+1)*16
                        img = F.pad(img, pad_dims, "constant")
                        dot_map = F.pad(dot_map, pad_dims, "constant")

                    assert img.size()[2:] == dot_map.size()[2:]

                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)

                            crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                            crop_gt.append(dot_map[:, :, h_start:h_end, w_start:w_end])
                            mask = torch.zeros_like(dot_map).cpu()
                            mask[:, :,h_start:h_end, w_start:w_end].fill_(1.0)
                            crop_masks.append(mask)
                    crop_imgs, crop_gt, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_gt, crop_masks))

                    # forward may need repeatng
                    crop_preds, crop_thresholds = [], []
                    nz, period = crop_imgs.size(0), 12
                    for i in range(0, nz, period):
                        [crop_threshold, crop_pred, __] = [i.cpu() for i in self.net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                        crop_preds.append(crop_pred)
                        crop_thresholds.append(crop_threshold)

                    crop_preds = torch.cat(crop_preds, dim=0)
                    crop_thresholds = torch.cat(crop_thresholds, dim=0)

                    # splice them to the original size
                    idx = 0
                    pred_map = torch.zeros_like(dot_map).cpu().float()
                    pred_threshold = torch.zeros_like(dot_map).cpu().float()
                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            pred_map[:, :, h_start:h_end, w_start:w_end]  += crop_preds[idx]
                            pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                            idx += 1

                # for the overlapping area, compute average value
                    mask = crop_masks.sum(dim=0)
                    pred_map = (pred_map / mask)
                    pred_threshold = (pred_threshold/mask)

                # binar_map = self.net.Binar(pred_map.cuda(), pred_threshold.cuda()).cpu()
                a = torch.ones_like(pred_map)
                b = torch.zeros_like(pred_map)
                binar_map = torch.where(pred_map >= pred_threshold, a, b)

                dot_map = dot_map.cpu()
                loss = F.mse_loss(pred_map, dot_map)

                losses.update(loss.item())
                binar_map = binar_map.numpy()
                pred_data,boxes = self.get_boxInfo_from_Binar_map(binar_map)
                # print(pred_data, gt_data)


                tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_metrics(num_classes,pred_data,gt_data)

                metrics_s['tp'].update(tp_s)
                metrics_s['fp'].update(fp_s)
                metrics_s['fn'].update(fn_s)
                metrics_s['tp_c'].update(tp_c_s)
                metrics_s['fn_c'].update(fn_c_s)
                metrics_l['tp'].update(tp_l)
                metrics_l['fp'].update(fp_l)
                metrics_l['fn'].update(fn_l)
                metrics_l['tp_c'].update(tp_c_l)
                metrics_l['fn_c'].update(fn_c_l)

  
                for c in range(len(tp_c_l)):
         
                    recall_c_list[c].append(tp_c_l[c] / (tp_c_l[c]+fn_c_l[c] + 1e-5))


                #    -----------Counting performance------------------
                gt_count, pred_cnt = gt_data['num'].numpy().astype(float), pred_data['num']
                s_mae = abs(gt_count - pred_cnt)
                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                cnt_errors['mae'].update(s_mae)
                cnt_errors['mse'].update(s_mse)
                if gt_count != 0:
                    s_nae = (abs(gt_count - pred_cnt) / gt_count)
                    cnt_errors['nae'].update(s_nae)

                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.tra_restore_transform, img,
                                pred_map.numpy(), dot_map.numpy(),binar_map,
                                pred_threshold.numpy(),boxes)

        ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
        ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
        f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s + 1e-20 )
        ar_c_s = metrics_s['tp_c'].sum / (metrics_s['tp_c'].sum + metrics_s['fn_c'].sum + 1e-20)

        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l+ 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)


        loss = losses.avg
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)

        self.writer.add_scalar('F1', f1m_l, self.epoch + 1)
        self.writer.add_scalar('Pre', ap_l, self.epoch + 1)
        self.writer.add_scalar('Rec', ar_l, self.epoch + 1)
       
        self.writer.add_scalar('overall_mae', mae, self.epoch + 1)
        self.writer.add_scalar('overall_mse', mse, self.epoch + 1)
        self.writer.add_scalar('overall_nae', nae, self.epoch + 1)

        self.train_record = update_model(self, [f1m_l, ap_l, ar_l,mae, mse, nae, loss])

        print_NWPU_summary(self,[f1m_l, ap_l, ar_l,mae, mse, nae, loss])
        content = f"{self.epoch}\t"
        for c in range(len(tp_c_l)):
            recall_c_list[c] = np.mean(recall_c_list[c])
            content += "{:.2f}".format(recall_c_list[c])
            content += '\t'
        content += '\n'
        # print(recall_c_list)
    
        with open('Class Recall.txt', 'a') as f:
            
            f.write(content)
        return f1m_l, ap_l, ar_l,mae, mse, nae