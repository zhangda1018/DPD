import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.locator import Crowd_locator
from model.HRSelector.HRSelector import *
from config import cfg
from misc.utils import *
from matplotlib import pyplot as plt
import datasets
import cv2, sys
from tqdm import tqdm
from misc.compute_metric import eval_metrics
from misc.EMA import EMA

class Trainer():
    def __init__(self, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.src_train_loader, self.src_val_loader, self.src_restore_transform = datasets.loading_data('SHHA')
        self.tra_train_loader, self.val_loader, self.tra_restore_transform = datasets.loading_data('SHHB')

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        if len(cfg.GPU_ID) > 1:
            self.selector = torch.nn.DataParallel(HRSelector(384)).cuda()
        else:
            self.selector = HRSelector(384).cuda()

        self.net = Crowd_locator(cfg.NET,cfg.GPU_ID,pretrained=True)
        state_dict = torch.load('/data2/Models/SHHA-VGG-ep_389_F1_0.664_Pre_0.729_Rec_0.610_mae_129.8_mse_278.7.pth')

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
                                         {'params':self.net.Binar.parameters(), 'lr':cfg.LR_BM_NET}])
            self.optimizer_selector = optim.AdamW(self.selector.parameters(), lr=3e-5)

        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.train_record = {'best_F1': 0, 'best_Pre': 0,'best_Rec': 0, 'best_mae': 1e20, 'best_mse':1e20, 'best_nae':1e20, 'best_model_name': ''}
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0
        self.num_iters = cfg.MAX_EPOCH * np.int(len(self.src_train_loader))
        self.rank_loss = nn.MarginRankingLoss()

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


    def forward(self):
        # self.validate()
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            self.epoch = epoch
            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if epoch%cfg.VAL_FREQ==0 and epoch>cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                self.validate()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )

            # if epoch > cfg.LR_DECAY_START:
            #     self.scheduler.step()

    def get_loss(self, pred_p, pred_b, gt):
        return F.mse_loss(pred_p, gt) + torch.abs(pred_b - gt).mean()

    def visual(self, img, prob_map):
        img = img[0]
        prob_map = prob_map[0]

        img = self.src_restore_transform(img.detach().cpu())
        img.save('test_img.jpg')
        plt.figure()
        plt.imshow(prob_map[0].detach().cpu().numpy())
        plt.jet()
        plt.colorbar()
        plt.savefig('Prob_map.jpg')
        plt.show()


    def train(self): # training for all datasets
        self.net.train()

        for iteration, data in enumerate(self.src_train_loader, 0):
            tra_data = iter(self.src_train_loader).next()
            self.i_tb+=1
            self.timer['iter time'].tic()
            img, strong_img, gt_map = data
            tra_img, tra_strong_img, tra_gt = tra_data
            
            tra_img = Variable(tra_img).cuda()
            tra_strong_img = Variable(tra_strong_img).cuda()
            # tra_gt = Variable(tra_gt).cuda()


            img = Variable(img).cuda()
            # strong_img = Variable(strong_img).cuda()
            gt_map = Variable(gt_map).cuda()
            batch_size = img.size(0)
            Ori_H, Ori_W = img.size()[2:]

            #Train Source
            while True:
                with torch.no_grad():
                    threshold_matrix, pre_map, binar_map = self.net(img, mask_gt = None, mode='val')  
                    sup_loss = self.get_loss(pre_map, binar_map, gt_map)                  
                    off_threshold_matrix, off_pre_map, off_binar_map = threshold_matrix.detach(), pre_map.detach(), binar_map.detach()
                prob_map, pred_scale = self.selector(torch.cat((off_threshold_matrix, off_pre_map), dim=1))
                break
            
            while True:
                with torch.no_grad():
                    biggest_factor = np.max(pred_scale.detach().cpu().numpy())
                    BIG_H, BIG_W = F.interpolate(img[0].unsqueeze(0), scale_factor=biggest_factor).size()[2:]
                    if BIG_H % 16 != 0:
                        BIG_H = (int(BIG_H  / 16) + 1) * 16
                    if BIG_W % 16 != 0:
                        BIG_W = (int(BIG_W  / 16) + 1) * 16
                    assert BIG_H % 16 == 0 and BIG_W % 16 == 0
                    scaled_imgs = []
                    effect_locs = []
                    for i in range(pred_scale.size(0)):
                        scaled_img = F.interpolate(img[i].unsqueeze(0), scale_factor=pred_scale[i].detach().cpu().item(), mode='bilinear')
  
                        tmp_H, tmp_W = scaled_img.size()[2:]
                        effect_locs.append([tmp_H, tmp_W])
                        scaled_img = F.pad(scaled_img, (0, BIG_H-tmp_H, 0, BIG_W-tmp_W), 'constant', 0)
                      
                        scaled_imgs.append(scaled_img)
                scaled_imgs = torch.cat(scaled_imgs, dim=0)
                scaled_imgs.requires_grad = True
                scaled_t_mat, scaled_p_map, scaled_b_map = self.net(scaled_imgs, mask_gt=None, mode='val')
                rescaled_p_map = []
                for i in range(scaled_p_map.size(0)):
                    tmp_H, tmp_W = effect_locs[i]
                    rescaled_p_map.append(F.interpolate(scaled_p_map[i, :, :tmp_H, :tmp_W].unsqueeze(0), size=(Ori_H, Ori_W)) * prob_map[i][0].unsqueeze(0).unsqueeze(0) + \
                                        pre_map[i].unsqueeze(0) * prob_map[i][1].unsqueeze(0).unsqueeze(0))
           

                rescaled_p_map = torch.cat(rescaled_p_map, dim=0)
                # scaled_loss = self.get_loss(scaled_p_map, scaled_b_map, gt_map) 
                scaled_loss = F.mse_loss(rescaled_p_map, gt_map)
                # rank_loss = self.rank_loss(sup_loss.unsqueeze(0), scaled_loss.unsqueeze(0), torch.ones_like(sup_loss.unsqueeze(0)).cuda())

                break
            

            while True: # Train Target
                tra_on_threshold_matrix, tra_on_pre_map, tra_on_binar_map = self.net(tra_strong_img, mask_gt = None, mode='val')                    


                with torch.no_grad():
                    self.ema.apply_shadow()
                    tra_off_threshold_matrix, tra_off_pre_map, tra_off_binar_map = self.net(tra_img, mask_gt = None, mode='val')                    
                    tra_off_threshold_matrix, tra_off_pre_map, tra_off_binar_map = tra_off_threshold_matrix.detach(), tra_off_pre_map.detach(), tra_off_binar_map.detach()
                    tra_prob_map, tra_pred_scale = self.selector(torch.cat((tra_off_threshold_matrix, tra_off_pre_map), dim=1))
                    tra_biggest_factor = np.max(tra_pred_scale.detach().cpu().numpy())
                    BIG_H, BIG_W = F.interpolate(tra_img[0].unsqueeze(0), scale_factor=tra_biggest_factor).size()[2:]
                    if BIG_H % 16 != 0:
                        BIG_H = (int(BIG_H  / 16) + 1) * 16
                    if BIG_W % 16 != 0:
                        BIG_W = (int(BIG_W  / 16) + 1) * 16
                    assert BIG_H % 16 == 0 and BIG_W % 16 == 0
                    tra_scaled_imgs = []
                    tra_effect_locs = []
                    for i in range(tra_pred_scale.size(0)):
                        tra_scaled_img = F.interpolate(tra_img[i].unsqueeze(0), scale_factor=tra_pred_scale[i].detach().cpu().item(), mode='bilinear')
  
                        tmp_H, tmp_W = scaled_img.size()[2:]
                        tra_effect_locs.append([tmp_H, tmp_W])
                        tra_scaled_img = F.pad(tra_scaled_img, (0, BIG_H-tmp_H, 0, BIG_W-tmp_W), 'constant', 0)
                      
                        tra_scaled_imgs.append(tra_scaled_img)
                    tra_scaled_imgs = torch.cat(tra_scaled_imgs, dim=0)
                

                

                
                    tra_scaled_t_mat, tra_scaled_p_map, tra_scaled_b_map = self.net(tra_scaled_imgs, mask_gt=None, mode='val')
                    tra_rescaled_p_map = []
                    tra_rescaled_t_mat = []
                    for i in range(tra_scaled_p_map.size(0)):
                        tmp_H, tmp_W = tra_effect_locs[i]
                        tra_rescaled_p_map.append(F.interpolate(tra_scaled_p_map[i, :, :tmp_H, :tmp_W].unsqueeze(0), size=(Ori_H, Ori_W)) * tra_prob_map[i][0].unsqueeze(0).unsqueeze(0) + \
                                            tra_off_pre_map[i].unsqueeze(0) * tra_prob_map[i][1].unsqueeze(0).unsqueeze(0))

                        tra_rescaled_t_mat.append(F.interpolate(tra_scaled_t_mat[i, :, :tmp_H, :tmp_W].unsqueeze(0), size=(Ori_H, Ori_W)) * tra_prob_map[i][0].unsqueeze(0).unsqueeze(0) + \
                                            tra_off_threshold_matrix[i].unsqueeze(0) * tra_prob_map[i][1].unsqueeze(0).unsqueeze(0))

                    tra_rescaled_p_map = torch.cat(tra_rescaled_p_map, dim=0)
                    tra_rescaled_t_mat = torch.cat(tra_rescaled_t_mat, dim=0)
                    tra_rescaled_mask = torch.where(tra_rescaled_p_map >= tra_rescaled_t_mat, 
                                        torch.ones_like(tra_rescaled_p_map).cuda(), torch.zeros_like(tra_rescaled_p_map).cuda())

                    self.ema.restore()
                break

            self.visual(img, prob_map)
            consis_loss = F.mse_loss(tra_on_pre_map, tra_rescaled_p_map) + torch.abs(tra_on_binar_map - tra_rescaled_mask).mean()

            all_loss =  scaled_loss + consis_loss
            


            self.optimizer.zero_grad()
            self.optimizer_selector.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            self.optimizer_selector.step()

            lr1,lr2 = adjust_learning_rate(self.optimizer,
                            cfg.LR_BASE_NET,
                            cfg.LR_BM_NET,
                            self.num_iters,
                            self.i_tb)
            sys.stdout.write('\r')
            sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d] Loss: [%.4f, %.4f | %.4f] Thre: [%.3f|%.3f] Factor: [%.3f]'
                    %(self.epoch + 1, 600, iteration + 1, len(self.src_train_loader), scaled_loss.item(), consis_loss.item(), scaled_loss.item(),
                    threshold_matrix.mean().item(), threshold_matrix.mean().item(), pred_scale.mean().item()))
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
            
            # if  i %100==0:
            #     box_pre, boxes = self.get_boxInfo_from_Binar_map(binar_map[0].detach().cpu().numpy())
            #     vis_results('tmp_vis', 0, self.writer, self.tra_restore_transform, tra_img, tra_pre_map[0].detach().cpu().numpy(), \
            #                      tra_gt[0].detach().cpu().numpy(),tra_binar_map.detach().cpu().numpy(),
            #                      tra_threshold_matrix.detach().cpu().numpy(),boxes)

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

    def validate(self):
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
        gen_tqdm = tqdm(self.val_loader)
        for vi, data in enumerate(gen_tqdm, 0):
            img,dot_map, gt_data = data
            slice_h, slice_w = self.cfg_data.TRAIN_SIZE

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
                    nz, period = crop_imgs.size(0), self.cfg_data.TRAIN_BATCH_SIZE
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