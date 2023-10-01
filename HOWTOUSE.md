# How to run PySlowFast Extractor

## C2D

```python
python run_test.py --cfg "./configs/C2D_NOPOOL_8x8_R50.yaml"
```

## I3D

```python
python run_test.py --cfg "./configs/I3D_8x8_R50.yaml"
```
1245mb


```python
python run_test.py --cfg "./configs/I3D_NLN_8x8_R50.yaml"
```
1271mb

```python
python run_test.py --cfg "./configs/I3D_NLN_8x8_R50_AN.yaml"
```

```python
python multiple_runs.py --cfg "./configs/I3D_8x8_R50.yaml"
```

## Slow

```python
python run_test.py --cfg "./configs/SLOW_8x8_R50.yaml"
```

```python
python run_test.py --cfg "./configs/SLOW_8x8_R50_AN.yaml"
```

1321mb

## SlowFast

```python
python run_test.py --cfg "./configs/SLOWFAST_8x8_R50_ALL.yaml"
```
1811mb

## SlowFast (charades)

```python
python run_test.py --cfg "./configs/SLOWFAST_16x8_R50.yaml"
```
imeza(1293M) 
```python
python run_test.py --cfg "./configs/SLOWFAST_16x8_R50_multigrid.yaml"
```
imeza(1251M)

```python
python run_test.py --cfg "./configs/SLOWFAST_16x8_R50_SSV2.yaml"
```

```python
python run_test.py --cfg "./configs/SLOWFAST_16x8_R101_AVA.yaml"
```

## X3D

```python
python run_test.py --cfg "./configs/X3D_M.yaml"
```

```python
python run_test.py --cfg "./configs/X3D_S.yaml"
```
1699mb
## MVit

```python
python run_test.py --cfg "./configs/MVIT_B_32x3_CONV_ALL.yaml"
```

```python
python run_test.py --cfg "./configs/MVITv2_S_16x4.yaml"
```
2293mb

```python
python run_test.py --cfg "./configs/MVITv2_S_16x4_AN.yaml"
```



```python
python multiple_runs.py --cfg "./configs/MVIT_B_32x3_CONV_ALL.yaml"
```

---

charades.py

```python
python charades_test.py --cfg "./configs/MVIT_B_32x3_CONV_ALL.yaml"
```

scp -P 202 31minutos.mp4 imeza@gate.dcc.uchile.cl:/data/imeza/ActivityNet/ActivityNet-apophis/2019/ActivityNetVideoData/v1-3/trial

Caso a comprobar
v_j7Tk8I_DCtw

# How to run Temporal Features

For TSN model ([Pretrained model link](https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth)):
```shell
python feature_extractor_2.py kinetics --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth --test_segments=5 --test_crops=1     --batch_size=1 --root_path=/data/imeza/youcookii/without_subfolders/videos/ --num_frame=8 --output_path=/data/imeza/youcookii/ycii_features/TSN_TEST/
```

For TSM model ([Pretrained model link](https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth)):

```shell
python feature_extractor_2.py kinetics --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth --test_segments=8 --test_crops=1     --batch_size=1 --root_path=/data/imeza/youcookii/without_subfolders/videos/ --num_frame=8 --output_path=/data/imeza/youcookii/ycii_features/TSM_TEST
```
