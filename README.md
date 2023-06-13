# BiconNets
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
====
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu

Recently accepted by Pattern Recognition.

Paper at: https://arxiv.org/abs/2103.00334

--------------------------------------------------------------------------

Updates:
06/12/2023: Major bugs observed. Will fix soon.
----
 
 
Requirement: Pytorch 1.7.1

This code including three parts:
1) Codes for customizing BiconNet wtih other backbones (/general)
2) Codes for reproducing the paper results (/paper_result)
3) Evaluation Code (/evaluation)

Customize the BiconNet based on your own network. (/general)
----

If you want to construct the BiconNet based on your own network, there are four simple steps:

1) replace your network's one-channel output fully connected layers with 8-channel FC layers.

For training:

2) generate the ground truth connectivity masks using the function 'sal2conn' in utils_bicon.py

3) replace your own loss function with Bicon_loss: you can edit the connect_loss.py

For testing:

4) use the function 'bv_test' in utils_bicon.py after you get the 8-channel connectivity map output to get your final saliency prediction.


Reproduce the results in the paper (/paper_result)
----------------------

For traing:
	cd /MODEL_NAME/bicon/train
	python train.py

For testing:
	cd /MODEL_NAME/bicon/test
	python test.py

The pretrained models and maps can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1rHcOnsgDt--K1hEidlILP3CCqih7cpgI?usp=sharing)


Results evaluation (/evaluation)
-------------
We use Matlab to evaluate the output saliency maps as did in: https://github.com/JosephChenHub/GCPANet


Citation
-------------
If you find this work useful in your research, please consider citing:
 
"Z. Yang, S. Soltanian-Zadeh, and S. Farsiu, "BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection", Pattern Recognition 121, 108231 (2022)"
