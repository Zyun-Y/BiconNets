# BiconNets
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection

Requirement: Pytorch 1.7.1

This code including three parts:
1) Codes for customizing BiconNet wtih other backbones (/general)
2) Codes for reproducing the paper results (/paper_result)
3) Evaluation Code (/evaluation)

1. Customize the BiconNet based on your own network. (/general)
If you want to construct the BiconNet based on your own network, there are four simple steps:

1) replace your network's one-channel output fully connected layers with 8-channel FC layers.

For training:
2) generate the ground truth connectivity masks using the function 'sal2conn' in utils_bicon.py
3) replace your own loss function with Bicon_loss: you can edit the connect_loss.py

For testing:
4) use the function 'bv_test' in utils_bicon.py after you get the 8-channel connectivity map output to get your final saliency prediction.
2. Customize the BiconNet based on your own network. (/general)
If you want to construct the BiconNet based on your own network, there are four simple steps:

1) replace your network's one-channel output fully connected layers with 8-channel FC layers.

For training:
2) generate the ground truth connectivity masks using the function 'sal2conn' in utils_bicon.py
3) replace your own loss function with Bicon_loss: you can edit the connect_loss.py

For testing:
4) use the function 'bv_test' in utils_bicon.py after you get the 8-channel connectivity map output to get your final saliency prediction.

------------------------------------------------------------------------------------------------------------------------------------------------------
2. Reproduce the results in the paper (/paper_result)

(a) PoolNet
Baseline code from: https://github.com/backseason/PoolNet

For traing:
	cd /PoolNet/bicon/train
	python train.py

For testing:
	cd /PoolNet/bicon/test
	python test.py

make sure the datapath is correct.

(b) CPD-R
Baseline code from: https://github.com/wuzhe71/CPD

For training
	cd /CPD-R/bicon/train
	python train.py

For testing:
	cd /CPD-R/bicon/test
	python test.py

make sure the datapath is correct.


(c) EGNet
Baseline code from: https://github.com/JXingZhao/EGNet

For training
	cd /EGNet/bicon/train
	python run.py

For testing:
	cd /EGNet/bicon/test
	python test.py

make sure the datapath is correct.

(d) GCPANet
Baseline code from: https://github.com/JosephChenHub/GCPANet

For training
	cd /GCPANet/bicon/train
	python train.py

For testing:
	cd /GCPANet/bicon/test
	python test.py

make sure the datapath is correct.

(d) ITSD
Baseline code from: https://github.com/moothes/ITSD-pytorch

For training
	cd /ITSD/bicon/train
	python train.py

For testing:
	cd /ITSD/bicon/test
	python test.py

make sure the datapath is correct.

(d) MINet
Baseline code from: https://github.com/lartpang/MINet

For training
	cd /MINet/bicon/train
	python main.py

For testing:
	cd /MINet/bicon/test
	python main.py

make sure the datapath is correct.


------------------------------------------------------------------------------------------------------------------------------------------------------

3. Results evaluation (/evaluation)
We use Matlab to evaluate the output saliency maps as did in: https://github.com/JosephChenHub/GCPANet

The pretrained models and maps can be downloaded at: https://drive.google.com/drive/folders/1rHcOnsgDt--K1hEidlILP3CCqih7cpgI?usp=sharing
