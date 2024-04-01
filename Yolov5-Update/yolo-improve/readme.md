# YOLO-Improve
这个项目主要是提供一些关于yolo系列模型的改进思路，效果因数据集和参数而异，仅作参考。  


# Explanation
- **iou**  
    添加EIOU，SIOU，ALPHA-IOU, FocalEIOU, Wise-IOU到yolov5,yolov8的box_iou中.  
- **yolov5-GFPN**   
    使用DAMO-YOLO中的GFPN替换YOLOV5中的Head.  
- **yolov5-C2F**  
    使用yolov8中的C2F模块替换yolov5中的C3模块.
- **yolov5-OTA**  
    添加Optimal Transport Assignment到yolov5的Loss中.  
- **yolov5-DCN**  
    添加Deformable convolution V2到yolov5中.  
- **yolov5-AUX**
    添加辅助训练分支到yolov5中.  
    原理参考链接：[知乎](https://zhuanlan.zhihu.com/p/588947172)
- **CAM**  
    添加context augmentation module到yolov5中.  
    paper：[链接](https://openreview.net/pdf?id=q2ZaVU6bEsT)
- **yolov5-SAConv**  
    添加SAC到yolov5中.  
    paper：[链接](https://arxiv.org/pdf/2006.02334.pdf)  
    reference: [链接](https://github.com/joe-siyuan-qiao/DetectoRS)
- **yolov5-CoordConv**  
    添加CoordConv到yolov5中.  
    reference: [链接](https://blog.csdn.net/qq_35608277/article/details/125257225)
- **yolov5-soft-nms**  
    添加soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU)到yolov5中.  
- **yolov5-DSConv**  
    添加DSConv到yolov5中.  
    paper: [链接](https://arxiv.org/abs/1901.01928)  
    reference: [链接](https://github.com/ActiveVisionLab/DSConv)
- **yolov5-DCNV3**  
    添加DCNV3到yolov5中.  
    paper: [链接](https://arxiv.org/abs/2211.05778)  
    reference: [链接](https://github.com/OpenGVLab/InternImage)  
- **yolov5-NWD**  
    添加Normalized Gaussian Wasserstein Distance到yolov5中.  
    paper: [链接](https://arxiv.org/abs/2110.13389)  
    reference: [链接](https://github.com/jwwangchn/NWD)  
- **yolov5-DecoupledHead**  
    添加Efficient-DecoupledHead到yolov5中.  
    paper: [yolov6链接](https://arxiv.org/pdf/2301.05586.pdf)  
    reference: [链接](https://github.com/meituan/YOLOv6/blob/main/yolov6/models/effidehead.py) 
- **yolov5-FasterBlock**  
    添加FasterNet中的Faster-Block到yolov5中.  
    paper: [链接](https://arxiv.org/abs/2303.03667)  
    reference: [链接](https://github.com/JierunChen/FasterNet) 
- **yolov5-backbone**  
    添加Timm支持的主干到yolov5中.  
    需要安装timm库. 命令: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm  
    reference: [链接](https://github.com/huggingface/pytorch-image-models#:~:text=I%20missed%20anything.-,Models,-All%20model%20architecture)
- **yolov5-TSCODE**  
    添加Task-Specific Context Decoupling到yolov5中.  
    需要安装einops库. 命令: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple einops  
    paper: [yolov6链接](https://arxiv.org/pdf/2303.01047v1.pdf)  
- **yolov5-backbone/fasternet**  
    添加FasterNet主干到yolov5中.  
    reference: [链接](https://github.com/JierunChen/FasterNet)
- **yolov5-backbone/ODConv**  
    添加Omni-Dimensional Dynamic Convolution主干(od_mobilenetv2,od_resnet)到yolov5中.  
    reference: [链接](https://github.com/OSVAI/ODConv)  
- **yolov5-backbone/ODConvFuse**  
    融合Omni-Dimensional Dynamic Convolution主干(od_mobilenetv2,od_resnet)中的Conv和BN.  
- **yolov5-CARAFE**  
    添加轻量级上采样算子CARAFE到yolov5中.  
    reference: [链接](https://github.com/XiaLiPKU/CARAFE)  
- **yolov5-EVC**  
    添加CFPNet中的EVC-Block到yolov5中.  
    reference: [链接](https://github.com/QY1994-0919/CFPNet)  
- **yolov5-dyhead**  
    添加基于注意力机制的目标检测头(DYHEAD)到yolov5中.  
    安装命令:

        pip install -U openmim
        mim install mmengine
        mim install "mmcv>=2.0.0"
    reference: [链接](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/dyhead.py)  
    paper: [链接](https://arxiv.org/abs/2106.08322)  
- **yolov5-backbone/inceptionnext**  
    添加(2023年New)InceptionNeXt主干到yolov5中.  
    reference: [链接](https://github.com/sail-sg/inceptionnext)  
    paper: [链接](https://arxiv.org/pdf/2303.16900.pdf)  
- **yolov5-aLRPLoss**  
    添加aLRPLoss到yolov5中.  
    reference: [链接](https://github.com/kemaloksuz/aLRPLoss)  
    paper: [链接](https://arxiv.org/abs/2009.13592)  
- **yolov5-res2block**  
    结合Res2Net提出具有多尺度提取能力的C3模块.  
    reference: [链接](https://github.com/Res2Net/Res2Net-PretrainedModels)  
    paper: [链接](https://arxiv.org/pdf/1904.01169.pdf)  
- **yolov5-backbone/FocalNet**  
    添加(2022年)FocalNet(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/microsoft/FocalNet)  
    paper: [链接](https://arxiv.org/abs/2203.11926)  
- **yolov5-backbone/EMO**  
    添加(2023年)EMO(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/zhangzjn/EMO)  
    paper: [链接](https://arxiv.org/pdf/2301.01146.pdf)  
- **yolov5-backbone/EfficientFormerV2**  
    添加(2022年)EfficientFormerV2(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/snap-research/EfficientFormer)  
    paper: [链接](https://arxiv.org/pdf/2212.08059.pdf)  
    weight_download: [百度网盘链接](https://pan.baidu.com/s/1I0Ygc3-6fNf2LdIJe290kw?pwd=yvc8)
- **yolov5-backbone/PoolFormer**  
    添加(2022年CVPR)PoolFormer(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/sail-sg/poolformer)  
    paper: [链接](https://arxiv.org/abs/2111.11418)  
- **yolov5-backbone/EfficientViT**  
    添加(2023年)EfficientViT(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/mit-han-lab/efficientvit)  
    paper: [链接](https://arxiv.org/abs/2205.14756)  
    weight_download: [百度网盘链接](https://pan.baidu.com/s/1dvwuQQBnRCr7aGReY8IEVw?pwd=74ad)
- **yolov5-ContextAggregation**  
    添加ContextAggregation到yolov5中.  
    reference: [链接](https://github.com/yeliudev/CATNet)  
    paper: [链接](https://arxiv.org/abs/2111.11057)  
- **yolov5-backbone/VanillaNet**  
    添加(2023年)VanillaNet主干到yolov5中.  
    reference: [链接](https://github.com/huawei-noah/VanillaNet)  
    paper: [链接](https://arxiv.org/abs/2305.12972)  
    weight_download: [百度网盘链接](https://pan.baidu.com/s/1EBAiOtDVMhvQqu2NWoFSIg?pwd=ofx9)  
- **yolov5-SwinTransformer**  
    添加SwinTransformer-Tiny主干到yolov5中.  
    reference: [链接](https://github.com/microsoft/Swin-Transformer)  
    weight_download: [SwinTransformer-Tiny百度云链接](https://pan.baidu.com/s/1vct0VYwwQQ8PYkBjwSSBZQ?pwd=swin)  
- **yolov5-NextViT**  
    添加(2022年)NextViT主干到yolov5中.  
    reference: [链接](https://github.com/bytedance/Next-ViT)  
    weight_download: [百度云链接](https://pan.baidu.com/s/18IHKssf9kN8Ej7zIWBKfcw?pwd=houj)  
- **yolov5-ConvNextV2**  
    添加(2023年)ConvNextV2主干到yolov5中.  
    reference: [链接](https://github.com/facebookresearch/ConvNeXt-V2)  
- **yolov5-RIFormer**  
    添加(2023年)RIFormer主干到yolov5中.  
    reference: [mmpretrain链接](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/riformer.py)  
    weight_download: [mmpretrain链接](https://github.com/open-mmlab/mmpretrain/tree/main/configs/riformer)
- **yolov5-C3RFEM**  
    Scale-Aware RFE与C3结合而成的C3RFEM添加到yolov5中.  
    reference: [链接](https://github.com/Krasjet-Yu/YOLO-FaceV2)  
- **yolov5-DBB**  
    把重参数结构DiverseBranchBlock与C3融合成C3-DBB添加到yolov5中.  
    reference: [链接](https://github.com/DingXiaoH/DiverseBranchBlock)  
- **yolov5-backbone/CVPR2023-EfficientViT**  
    添加(2023CVPR)EfficientViT(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/microsoft/Cream/tree/main/EfficientViT)  
    paper: [链接](https://arxiv.org/pdf/2305.07027.pdf)  
    weight: [github链接](https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/tag/v1.0)
- **yolov5-backbone/LSKNet**  
    添加(2023旋转目标检测SOTA)LSKNet主干到yolov5中.  
    reference: [链接](https://github.com/zcablii/LSKNet)  
    paper: [链接](https://arxiv.org/pdf/2303.09030.pdf)  
- **yolov5-MPDiou**  
    添加(2023最新IoU度量算法)MPDiou到yolov5中.(视频教学地址中为详细从头手把手教学,因此本项没有提供代码)  
    paper: [链接](https://arxiv.org/pdf/2307.07662v1.pdf)  
- **yolov5-SlideLoss**  
    添加Yolo-Face-V2中SlideLoss的到yolov5中.  
    reference: [链接](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)  
    paper: [链接](https://arxiv.org/abs/2208.02019)  
- **yolov5-backbone/CVPR2023-RepViT**  
    添加RepViT(transformer)主干到yolov5中.  
    reference: [链接](https://github.com/THU-MIG/RepViT)  
    paper: [链接](https://arxiv.org/abs/2307.09283)  
- **yolov5-GOLDYOLO**  
    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进YOLOV5中的特征融合模块.  
    reference: [链接](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)  
    paper: [链接](https://arxiv.org/abs/2309.11331)  
- **yolov5-DySnakeConv**  
    利用动态蛇形卷积改进YOLOV5.  
    reference: [链接](https://github.com/YaoleiQi/DSCNet)  
    paper: [链接](https://arxiv.org/abs/2307.08388)  
- **yolov5-AIFI**  
    利用带有位置信息编码的AIFI自注意力机制改进YOLOV5.  
    reference: [链接](https://github.com/lyuwenyu/RT-DETR)  
    paper: [链接](https://arxiv.org/pdf/2304.08069.pdf)   
- **yolov5-backbone/UniRepLKNet**  
    添加UniRepLKNet主干到yolov5中.  
    reference: [链接](https://github.com/AILab-CVC/UniRepLKNet)  
    paper: [链接](https://arxiv.org/abs/2311.15599)  
    weights-download: [百度云链接](https://pan.baidu.com/s/1Gk48Xa6cWKAVJgsF5cqk1g?pwd=b55v)
- **yolov5-asf** 
    添加Attentional Scale Sequence Fusion到yolov5中.
    reference: [链接](https://github.com/mkang315/ASF-YOLO)  
    paper: [链接](https://arxiv.org/abs/2312.06458)  
- **yolov5-ccfm**
    添加cross-scale feature-fusion到yolov5中.
    reference: [链接](https://github.com/ultralytics/ultralytics)  
    paper: [链接](https://arxiv.org/pdf/2304.08069.pdf)  
- **yolov5-RepNCSPELAN**
    添加yolov9中的RepNCSPELAN到yolov5中.
    reference: [链接](https://github.com/WongKinYiu/yolov9)  
    paper: [链接](https://arxiv.org/abs/2402.13616)
