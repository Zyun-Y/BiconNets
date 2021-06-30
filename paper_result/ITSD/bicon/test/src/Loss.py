import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable


class FS_loss(nn.Module):
  def __init__(self, weights, b=0.3):
    super(FS_loss, self).__init__()
    self.contour = weights
    self.b = b

  def forward(self, X, Y, weights):
    loss = 0
    batch = Y.size(0)
    
    for weight, x in zip(weights, X):
        pre = x.sigmoid_()
        scale = int(Y.size(2) / x.size(2))
        pos = F.avg_pool2d(Y, kernel_size=scale, stride=scale).gt(0.5).float()
        tp = pre * pos
        
        tp = (tp.view(batch, -1)).sum(dim = -1)
        posi = (pos.view(batch, -1)).sum(dim = -1)
        pre = (pre.view(batch, -1)).sum(dim = -1)
        
        f_score = tp * (1 + self.b) / (self.b * posi + pre)
        loss += weight * (1 - f_score.mean())
    return loss



def edge_loss(vote_out,edge):
    pred_mask_min, _ = torch.min(vote_out, dim=1)
    pred_mask_max,_ = torch.max(vote_out, dim=1)
    pred_mask_min = pred_mask_min * edge
    minloss = F.binary_cross_entropy(pred_mask_min,torch.full_like(pred_mask_min, 0),reduction='sum')
    return minloss

def conn_loss(c_map,target,con_target):
    bce = nn.BCELoss()
    con_target = con_target.type(torch.FloatTensor).cuda()
    hori_translation = torch.zeros([c_map.shape[0],c_map.shape[3],c_map.shape[3]])
    for i in range(c_map.shape[3]-1):
        hori_translation[:,i,i+1] = torch.tensor(1.0)
    verti_translation = torch.zeros([c_map.shape[0],c_map.shape[2],c_map.shape[2]])
    for j in range(c_map.shape[2]-1):
        verti_translation[:,j,j+1] = torch.tensor(1.0)
    hori_translation = hori_translation.float()
    verti_translation = verti_translation.float()

    sum_conn = torch.sum(con_target,dim=1)
    edge = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
    edge0 = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
    edge = edge*edge0

    target = target.type(torch.FloatTensor).cuda()
    c_map = F.sigmoid(c_map)

    pred_mask,vote_out = ConMap2Mask_prob(c_map,hori_translation,verti_translation)

    pred_mask_mean = torch.mean(vote_out, dim=1)
    pred_mask_mean[torch.where(edge==0)] = 0 #store edge


    # edge_loss1 = edge_loss(vote_out,edge)
    # non_edge_tar = target-edge.unsqueeze(1)
    # non_edge_pred = pred_mask-pred_mask_mean


    # bce_loss = bce(non_edge_pred.unsqueeze(1), non_edge_tar)
    conn_l = bce(c_map,con_target)
    bimap_l = bce(c_map,con_target)
    loss =  0.8*conn_l+0.2*bimap_l#+bce_loss + edge_loss1

    return pred_mask, loss

def ACT(X, batchs, args):
    bce = nn.BCELoss(reduction='none')

    slc_gt = torch.tensor(batchs['Y']).cuda()
    ctr_gt = torch.tensor(batchs['C']).cuda()
    # print(slc_gt.shape)
    conn_gt = connectivity_matrix(slc_gt.unsqueeze(1))
    slc_loss, ctr_loss = 0, 0
    for slc_pred, ctr_pred, weight in zip(X['preds'], X['contour'], args.weights):
        scale = int(slc_gt.size(-1) / slc_pred.size(-1))
        ys = F.avg_pool2d(slc_gt, kernel_size=scale, stride=scale).gt(0.5).float()
        yc = F.max_pool2d(ctr_gt, kernel_size=scale, stride=scale)
        yconn = connectivity_matrix(ys.unsqueeze(1))
        # print(slc_pred.s)
        slc_pred, slc_conn_l = conn_loss(slc_pred,ys.unsqueeze(1),yconn)

        # print(ys.shape, slc_pred.shape)

        ctr_pred = F.sigmoid(ctr_pred)


        # print(slc_pred.shape)
        # contour loss
        #w = torch.yc

        # ACT loss
        pc = ctr_pred.clone()
        w = torch.where(pc > yc, pc, yc)

        slc_loss += ((bce(slc_pred, ys) * (w * 4 + 1)).mean() * weight +slc_conn_l)
            
        if ctr_pred is not None:
            ctr_pred = ctr_pred.squeeze(1)
            ctr_loss += bce(ctr_pred, yc).mean() * weight

    # print(pc.shape)
    pc = F.interpolate(pc, size=ctr_gt.size()[-2:], mode='bilinear').squeeze(1)
    w = torch.where(pc > ctr_gt, pc, ctr_gt)

    X['final'] = F.sigmoid(X['final'])
    X['final'], final_conn_l = conn_loss(X['final'],slc_gt.unsqueeze(1),conn_gt)
    fnl_loss= (bce(X['final'], slc_gt.gt(0.5).float()) * (w * 4 + 1)).mean() * args.weights[-1]

    total_loss =  final_conn_l+ctr_loss + slc_loss + fnl_loss
    # print(total_loss)
    return total_loss





def ConMap2Mask_prob(c_map,hori_translation,verti_translation):

    
    hori_translation = hori_translation.cuda()
    # print(hori_translation)
    verti_translation = verti_translation.cuda()
    # print(hori_translation.shape)
    batch,channel, row, column = c_map.size()
    vote_out = torch.zeros([batch,channel, row, column]).cuda()

    eps = 0
    # print(c_map[1,4].shape)
    right = torch.bmm(c_map[:,4],hori_translation)
    left = torch.bmm(c_map[:,3],hori_translation.transpose(2,1))
    left_bottom = torch.bmm(verti_translation.transpose(2,1), c_map[:,5])
    left_bottom = torch.bmm(left_bottom,hori_translation.transpose(2,1))
    right_above = torch.bmm(verti_translation, c_map[:,2])
    right_above= torch.bmm(right_above,hori_translation)
    left_above = torch.bmm(verti_translation, c_map[:,0])
    left_above = torch.bmm(left_above,hori_translation.transpose(2,1))
    bottom = torch.bmm(verti_translation.transpose(2,1), c_map[:,6])
    up = torch.bmm(verti_translation, c_map[:,1])
    right_bottom = torch.bmm(verti_translation.transpose(2,1), c_map[:,7])
    right_bottom = torch.bmm(right_bottom,hori_translation)
    # print(right[0][0][100])
    # print(c_map[:,3][0][0][100])
    a1 = (c_map[:,3]) * (right)

    # print(a1[0][0][100])
    a2 = (c_map[:,4]) * (left)
    a3 = (c_map[:,1]) * (bottom)
    a4 = (c_map[:,6]) * (up+eps)
    a5 = (c_map[:,2]) * (left_bottom)
    a6 = (c_map[:,5]) * (right_above)
    a7 =(c_map[:,0]) * (right_bottom)
    a8 =(c_map[:,7]) * (left_above)
    vote_out[:,0] = a7
    vote_out[:,1] = a3
    vote_out[:,2] = a5
    vote_out[:,3] = a1
    vote_out[:,4] = a2
    vote_out[:,5] = a6
    vote_out[:,6] = a4
    vote_out[:,7] = a8
    # vote_out = vote_out.cuda()
    # pred_mask = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(a1,a2),a3),a4),a5),a6),a7),a8)
    pred_mask = torch.mean(vote_out,dim=1)
    # print(ored)
    # print(pred_mask[1])
    return pred_mask,vote_out


def connectivity_matrix(mask):
    # print(mask.shape)
    [batch,channels,rows, cols] = mask.shape

    conn = torch.zeros([batch,8,rows, cols]).cuda()
    up = torch.zeros([batch,rows, cols]).cuda()#move the orignal mask to up
    down = torch.zeros([batch,rows, cols]).cuda()
    left = torch.zeros([batch,rows, cols]).cuda()
    right = torch.zeros([batch,rows, cols]).cuda()
    up_left = torch.zeros([batch,rows, cols]).cuda()
    up_right = torch.zeros([batch,rows, cols]).cuda()
    down_left = torch.zeros([batch,rows, cols]).cuda()
    down_right = torch.zeros([batch,rows, cols]).cuda()


    up[:,:rows-1, :] = mask[:,0,1:rows,:]
    down[:,1:rows,:] = mask[:,0,0:rows-1,:]
    left[:,:,:cols-1] = mask[:,0,:,1:cols]
    right[:,:,1:cols] = mask[:,0,:,:cols-1]
    up_left[:,0:rows-1,0:cols-1] = mask[:,0,1:rows,1:cols]
    up_right[:,0:rows-1,1:cols] = mask[:,0,1:rows,0:cols-1]
    down_left[:,1:rows,0:cols-1] = mask[:,0,0:rows-1,1:cols]
    down_right[:,1:rows,1:cols] = mask[:,0,0:rows-1,0:cols-1]

    # print(mask.shape,down_right.shape)
    conn[:,0] = mask[:,0]*down_right
    conn[:,1] = mask[:,0]*down
    conn[:,2] = mask[:,0]*down_left
    conn[:,3] = mask[:,0]*right
    conn[:,4] = mask[:,0]*left
    conn[:,5] = mask[:,0]*up_right
    conn[:,6] = mask[:,0]*up
    conn[:,7] = mask[:,0]*up_left
    conn = conn.type(torch.FloatTensor).cuda()

    return conn