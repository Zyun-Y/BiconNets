import os
import time

import numpy as np
import torch
from torchvision import transforms
import scipy
import scipy.ndimage

class Eval_thread(): 
	def __init__(self):
		m = 0


	def run(self,pred,gt):
		mae = self.Eval_mae(pred,gt)
		fmax = self.Eval_fmeasure(pred,gt)
		# S = self.S_measure(pred,gt)
		# E = self.Eval_Emeasure(pred,gt)
		return mae, fmax

	def Eval_mae(self,pred,gt):
	    avg_mae, img_num = 0.0, 0.0
	    #mae_list = [] # for debug
	    with torch.no_grad():
	        mea = torch.abs(pred - gt).mean()

	    return mea.item()

	def Eval_fmeasure(self,pred,gt):

	    beta2 = 0.3
	    avg_f, img_num = 0.0, 0.0
	    score = torch.zeros(255)

	    with torch.no_grad():

	        prec, recall = self.eval_pr(pred, gt, 255)
	        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
	        f_score[f_score != f_score] = 0 # for Nan
	        avg_f += f_score
	        img_num += 1.0
	        score = avg_f / img_num

	    return score.max().item(), torch.mean(score).item()

	def Eval_Emeasure(self,pred,gt):
		avg_e = 0.0
		with torch.no_grad():
			scores = torch.zeros(255)
			scores = scores.cuda()

			scores += self._eval_e(pred, gt, 255)


		return scores.mean().item()

	def _eval_e(self, y_pred, y, num):
		y = y.float().cuda()
		y_pred = y_pred.cuda()
		score = torch.zeros(num).cuda()
		thlist = torch.linspace(0, 1 - 1e-10, num).cuda()

		for i in range(num):
			y_pred_th = (y_pred >= thlist[i]).float()
			if torch.mean(y.float()) == 0.0: # the ground-truth is totally black
				y_pred_th = torch.mul(y_pred_th, -1)
				enhanced = torch.add(y_pred_th, 1)
			elif torch.mean(y.float()) == 1.0: # the ground-truth is totally white
				enhanced = y_pred_th
			else: # normal cases
				fm = y_pred_th - y_pred_th.mean()
				gt = y - y.mean()
				align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
				enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

			score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

		return score

	def eval_pr(self, y_pred, y, num):

		prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
		thlist = torch.linspace(0, 1 - 1e-10, num).cuda()

		for i in range(num):
			y_temp = (y_pred >= thlist[i]).float()
			tp = (y_temp * y).sum()
			prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

		return prec, recall

	def _ssim(self, pred, gt):
		gt = gt.float()
		h, w = pred.size()[-2:]
		N = h*w
		x = pred.mean()
		y = gt.mean()
		sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
		sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
		sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)

		aplha = 4 * x * y *sigma_xy
		beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

		if aplha != 0:
		    Q = aplha / (beta + 1e-20)
		elif aplha == 0 and beta == 0:
		    Q = 1.0
		else:
		    Q = 0

		return Q

	def S_measure(self,pred,gt):
		alpha = 0.5
		avg_q =0
		# gt = gt.type(torch.LongTensor).cuda()
		y = torch.mean(gt.float())
		#y = gt.mean()
		if y == 0:
		    x = pred.mean()
		    Q = 1.0 - x
		elif y == 1:
		    x = pred.mean()
		    Q = x
		else:
		    gt[gt>=0.5] = 1
		    gt[gt<0.5] = 0
		    Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
		    if Q.item() < 0:
		        Q = torch.FloatTensor([0.0])
		avg_q += Q.item()

		return avg_q

	def _S_region(self, pred, gt):
		X, Y = self._centroid(gt)
		gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
		p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
		Q1 = self._ssim(p1, gt1)
		Q2 = self._ssim(p2, gt2)
		Q3 = self._ssim(p3, gt3)
		Q4 = self._ssim(p4, gt4)
		Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
		# print(Q)

		return Q

	def _S_object(self, pred, gt):
		gt = gt.float()
		fg = torch.where(gt==0, torch.zeros_like(pred), pred)
		bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
		o_fg = self._object(fg, gt)
		o_bg = self._object(bg, 1-gt)
		u = gt.mean()
		Q = u * o_fg + (1-u) * o_bg

		return Q

	def _object(self, pred, gt):
		temp = pred[gt == 1]
		x = temp.mean()
		sigma_x = temp.std()
		score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
		return score

	def _centroid(self, gt):
		rows, cols = gt.size()[-2:]
		gt = gt.view(rows, cols)
		if gt.sum() == 0:

			X = torch.eye(1).cuda() * round(cols / 2)
			Y = torch.eye(1).cuda() * round(rows / 2)

		else:
			total = gt.sum()

			i = torch.from_numpy(np.arange(0,cols)).cuda().float()
			j = torch.from_numpy(np.arange(0,rows)).cuda().float()

			X = torch.round((gt.sum(dim=0)*i).sum() / total)
			Y = torch.round((gt.sum(dim=1)*j).sum() / total)

		return X.long(), Y.long()
    
	def _divideGT(self, gt, X, Y):
		h, w = gt.size()[-2:]
		area = h*w
		gt = gt.view(h, w)
		LT = gt[:Y, :X]
		RT = gt[:Y, X:w]
		LB = gt[Y:h, :X]
		RB = gt[Y:h, X:w]
		X = X.float()
		Y = Y.float()
		w1 = X * Y / area
		w2 = (w - X) * Y / area
		w3 = X * (h - Y) / area
		w4 = 1 - w1 - w2 - w3

		return LT, RT, LB, RB, w1, w2, w3, w4

	def _dividePrediction(self, pred, X, Y):
		h, w = pred.size()[-2:]
		pred = pred.view(h, w)
		LT = pred[:Y, :X]
		RT = pred[:Y, X:w]
		LB = pred[Y:h, :X]
		RB = pred[Y:h, X:w]

		return LT, RT, LB, RB