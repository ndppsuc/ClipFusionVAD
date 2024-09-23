# ClipFusionVAD
ClipFusionVAD: Enhancing Weakly-Supervised Fine-Grained Multi-Class Video Anomaly Detection with Multi-Modal Feature Fusion
![framework](data/framework.png)



## Highlight

- In the video-text multimodal feature fusion stage, the Visual Aggregator module and the Visual Product module are introduced. Benefiting from the cooperation of the two modules, the complementary multimodal features are enhanced, and the discriminative ability of multi-modal features is improved, to achieve more accurate fine-grained anomaly detection.
- In the extraction of video features, global attention is proposed. By combining channel attention and spatial attention, the description capability of visual features is improved. The global attention mechanism can significantly improve the accuracy and robustness of anomaly detection, making the model valid in complex environments and improving the performance of video anomaly detection.
- The proposed ClipFusionVAD model outperforms other models in both binary anomaly detection tasks and fine-grained multi-classification tasks. In the binary classification task, ClipFusionVAD achieved 86.77% AUC on the UCF-Crime dataset and 77.78 % accuracy on the XD-Violence dataset. In the multi-classification task, the model significantly outperformed baseline methods, the average mAP on the two datasets reaches 9.4 and 28.07 respectively.



## abstract
Fine-grained multi-class video anomaly detection(VAD) is a research hotspot in computer vision. By extracting temporal and spatial features from videos， VAD predicts the binary results of whether there are abnormalities in the video, locates the anomaly area and happening time, and can also predict the fine-grained categories of abnormal events. 
To address the insufficient feature representation of existing models, we propose a dual-branch model named ClipFusionVAD for multi-modal feature fusion. On the one hand, by introducing a Global Attention Module(GAM) that combines spatial and channel attention, the discriminative ability of visual features is enhanced. On the other hand, a Visual Aggregator module and a Visual Product module are proposed to fuse the multi-modal features, improving the representation capabilities of multimodal features while further optimizing feature details,  so that ClipFusionVAD can effectively capture the key information in complex scenes and achieving more accurate fine-grained abnormal event classification.  ClipFusionVAD effectively captures key information in complex scenes, achieving more accurate fine-grained anomaly classification. Experimental results indicate that ClipFusionVAD achieved the AUC of 86.77%  on the UCF-Crime dataset and 77.78% on the XD-Violence dataset in binary anomaly detection. In the fine-grained multi-class anomaly classification task, our method significantly outperformed all baseline methods, with average mean Average Precision (mAP) scores on two datasets at 9.4 and 28.07 respectively.  Overall, ClipFusionVAD demonstrates outstanding performance in both binary and multi-class tasks, achieving state-of-the-art detection anomaly detection performance. The project page is available at https://github.com/ndppsuc/ClipFusionVAD


## Training
We extract CLIP features for UCF-Crime and XD-Violence datasets, and release these features as follows:
ucf-crime：
xd-violence：

- Change the file paths to the download datasets above in `list/xd_CLIP_rgb.csv` and `list/xd_CLIP_rgbtest.csv`. 
- Feel free to change the hyperparameters in `xd_option.py`

Traing and infer for XD-Violence dataset
```
python xd_train.py
python xd_test.py
```
Traing and infer for UCF-Crime dataset
```
python ucf_train.py
python ucf_test.py
```


## References
We referenced the repos below for the code.
* [VadClip](https://github.com/nwpu-zxr/VadCLIP)

