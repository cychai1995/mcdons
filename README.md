# Multi-class Dense Object Nets
### Code for our paper "Multi-step Pick-and-Place Tasks Using Object-centric Dense Correspondences" (IROS 2019)

## Training Environment
- OS: Ubuntu 18.04 with Python Version=3.6

- Device and Packages: using **single** GTX-1080Ti / RTX-2080Ti with **CUDA 10.1** and

```
torch==1.1.0
opencv-python==4.10.0.*
torchvision==0.2.2*
```

- [Install NVIDIA/apex package](https://github.com/NVIDIA/apex) for FP16 optimization.

```bash
git clone https://github.com/NVIDIA/apex
git checkout 8818ba9
python setup.py install --cuda_ext --cpp_ext
```

Please make sure the correct installation of pytorch(1.1.0) and apex, otherwise the memory usage may excess 11GB.

## Data Download

### Pretrained Feature-pyramid networks on PascalVOC 2007

- [Download fpn50.weight](https://drive.google.com/open?id=1ZrufPSS7LFSM1fRxL0Jp5mOcp8bZ3jgM)
- The weight file was extracted from the pretrained **FPNSSD512** model released from [TorchCV: a PyTorch vision library mimics ChainerCV](https://github.com/kuangliu/torchcv)

### Pascal VOC 2012 Images for background randomization

- [Download the VOC2012 Data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)( ```VOCtrainval_11-May-2012.tar```)

### Our Dataset

-  [Download DataMCD-net.zip](https://drive.google.com/file/d/1N6ZstxkK_tiXKqkPRePXQx878Oxn-lrt/view?usp=sharing)

## Modifications for the configuration files

Modify the following lines in the configuration file you are going to run, e.g., ```configs/mcdon.yaml```

```yaml
FPN_pretrained: 'YOUR_PATH_FPN50/fpn50.weight'
texture_base: 'YOUR_PATH_VOC2012/VOCdevkit/VOC2012'
progress_path: 'YOUR_PATH_CHECKPOINT/progress' 
result_path: 'YOUR_PATH_CHECKPOINT/result'
base_dir: 'YOUR_PATH_DATASET/DataMCD-net'
```

- YOUR_PATH_FPN50: the path for finding the pretrained weight file ```fpn50.weight```

- YOUR_PATH_VOC2012: the path where you extract the downloaded ```VOCtrainval_11-May-2012.tar```
- YOUR_PATH_CHECKPOINT: any desired path for the checkpoints during training
- YOUR_PATH_DATASET: the path where you extract our dataset file

## Run the training

- Train the **DON Baseline** on 7 classes of objects

  ```python trainer.py configs/baseline.yaml```

- Train our **MCDON** (MHSCT-12D M=5,N=2) on 7 classes of objects

  ```python trainer.py configs/mcdon.yaml```

## Run the visual evaluation(on novel objects) after training
```bash
python descriptor_evaluation configs/mcdon.yaml
```
- Pass ```-tps N``` to manually specify the restoring ```N``` epoch, otherwise we use the ```tps``` value in the configuration file
- In the evaluation window,
  - Adjust the proper threshold on the bar
  - Click on either side of the color images to find the matching region in the other side
  - Press ```s``` to sample new pairs for comparison, ```q``` to exit

## Pretrained Weights

- Pretrained weight file of our **MCDON** using the setting ```configs/mcdon.yaml```

  - [Download progress50.tpdata](https://drive.google.com/file/d/1_rzDrLpdSRguW5r4OgzdoVD949VKbiVb/view?usp=sharing)

  - Place the weight file in the path

    ``````
    YOUR_PATH_CHECKPOINT/progress/mcdon/progress50.tpdata
    ``````
  - Run the visual evaluation
    ``````
    python descriptor_evaluation.py configs/mcdon.yaml -tps 50
    ``````

