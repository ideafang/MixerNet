# MixerNet
Code for our CNSM'2022 paper "A Novel Network Delay Prediction Model with Mixed Multi-layer Perceptron Architecture for Edge Computing"

![](img/model.png)
If you find this code useful in your research please cite

```
@inproceedings{fang2022a,
  title={A Novel Network Delay Prediction Model with Mixed Multi-layer Perceptron Architecture for Edge Computing},
  author={Honglin Fang and Peng Yu and Ying Wang and Wenjing Li and Fanqin Zhou and Run Ma},
  booktitle={CNSM},
  year={2022},
}
```

## Setup


The main environment is (latest is also available):
* cuda 11.3
* torch 1.8.0
* networkx 2.6.3
* einops 0.4.0

### Prepare dataset
NSFNET and GEANT2 datasets are publicly available [here](https://github.com/BNN-UPC/NetworkModelingDatasets/tree/master/datasets_v0)

```
cd dataset
# For NSFNET
wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
tar -xvzf nsfnet.tar.gz 
# For GEANT2
wget "http://knowledgedefinednetworking.org/data/datasets_v0/geant2.tar.gz"
tar -xvzf geant2.tar.gz
```

## Train Model
* **Train with data process(first time)**
```
# for NSFNET
python run.py --net nsfnetbw
# for GEANT2
python run.py --net geant2bw
```
* **Train**
```
# for NSFNET
python run.py --net nsfnetbw --process False
# for GEANT2
python run.py --net geant2bw --process False
```


## Related publications:
* RouteNet (JSAC 2020) [paper](https://ieeexplore.ieee.org/document/9109574) [code](https://github.com/knowledgedefinednetworking/demo-routenet)