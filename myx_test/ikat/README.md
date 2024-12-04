# 数据集说明

`.pkl`格式存储，使用python的`pickle`模块进行加载。

加载后会得到一个字典

```python
{"data": List[List[List[float]]], "label": List[int]}
```

其中`data`为一个三维列表，第一维为trajectory的数量，第二维为每条trajectory的长度，第三维为每条trajectory的维度（2维）。

`label`为一个一维列表，表示每条trajectory的是否为异常，异常为1，正常为0。

## baboons
trajectory nums:       	2310
min trajectory length: 	30
max trajectory length: 	603
mean trajectory length:	441.60
trajectory dimension:  	2
anomaly nums:          	110
anomaly ratio:         	4.7619%

## CASIA
trajectory nums:       	1500
min trajectory length: 	16
max trajectory length: 	612
mean trajectory length:	95.59
trajectory dimension:  	2
anomaly nums:          	24
anomaly ratio:         	1.6000%

## curlew
trajectory nums:       	42
min trajectory length: 	488
max trajectory length: 	71821
mean trajectory length:	19083.05
trajectory dimension:  	2
anomaly nums:          	9
anomaly ratio:         	21.4286%

## Detrac
trajectory nums:       	5356
min trajectory length: 	11
max trajectory length: 	2120
mean trajectory length:	83.09
trajectory dimension:  	2
anomaly nums:          	71
anomaly ratio:         	1.3256%

## flyingfox
trajectory nums:       	62
min trajectory length: 	517
max trajectory length: 	4768
mean trajectory length:	2133.10
trajectory dimension:  	2
anomaly nums:          	11
anomaly ratio:         	17.7419%

## newsheepdogs
trajectory nums:       	538
min trajectory length: 	11
max trajectory length: 	3501
mean trajectory length:	121.88
trajectory dimension:  	2
anomaly nums:          	23
anomaly ratio:         	4.2751%

## pigeons
trajectory nums:       	58
min trajectory length: 	12
max trajectory length: 	4795
mean trajectory length:	570.88
trajectory dimension:  	2
anomaly nums:          	8
anomaly ratio:         	13.7931%

## sheepdogs
trajectory nums:       	603
min trajectory length: 	11
max trajectory length: 	3501
mean trajectory length:	110.13
trajectory dimension:  	2
anomaly nums:          	88
anomaly ratio:         	14.5937%

## turkey_vulture
trajectory nums:       	67
min trajectory length: 	172
max trajectory length: 	7721
mean trajectory length:	3171.42
trajectory dimension:  	2
anomaly nums:          	15
anomaly ratio:         	22.3881%

## VRU
trajectory nums:       	1168
min trajectory length: 	55
max trajectory length: 	1257
mean trajectory length:	325.99
trajectory dimension:  	2
anomaly nums:          	100
anomaly ratio:         	8.5616%

## wildebeest
trajectory nums:       	92
min trajectory length: 	138
max trajectory length: 	5672
mean trajectory length:	3033.50
trajectory dimension:  	2
anomaly nums:          	14
anomaly ratio:         	15.2174%

