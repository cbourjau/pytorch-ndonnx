## alexnet

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.8677 | 1.0433 |
| ONNX eager                 | 0.9140 | 1.2162 |
| ONNX infer from session    | 0.0312 | 0.1717 |
| ONNX jit                   | 1.1248 | 1.3460 |
| ONNX session creation      | 0.1739 | 0.2000 |
| pytorch compile            | 1.0912 | 0.6029 |
| pytorch eager:             | 0.0573 | 0.2122 |
| pytorch inference compiled | 0.0565 | 0.1988 |
| pytorch onnx export        | 1.3751 | 0.4013 |

## densenet

| Batch size                 | 2      | 10      |
| -------------------------- | ------ | ------- |
| ONNX build                 | 0.9406 | 0.8609  |
| ONNX eager                 | 4.5343 | 14.2444 |
| ONNX infer from session    | 0.1456 | 0.6946  |
| ONNX jit                   | 1.3067 | 1.7530  |
| ONNX session creation      | 0.1283 | 0.1793  |
| pytorch compile            | 6.5535 | 10.7388 |
| pytorch eager:             | 0.2071 | 0.9665  |
| pytorch inference compiled | 0.1692 | 0.9412  |
| pytorch onnx export        | nan    | nan     |

## googlenet

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.8008 | 0.5569 |
| ONNX eager                 | 1.7558 | 4.2156 |
| ONNX infer from session    | 0.0740 | 0.3621 |
| ONNX jit                   | 0.6796 | 0.9849 |
| ONNX session creation      | 0.0522 | 0.0662 |
| pytorch compile            | 3.0300 | 5.1584 |
| pytorch eager:             | 0.1141 | 0.6089 |
| pytorch inference compiled | 0.0844 | 0.4718 |
| pytorch onnx export        | nan    | nan    |

## mnasnet

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.3802 | 0.3801 |
| ONNX eager                 | 1.3740 | 5.2448 |
| ONNX infer from session    | 0.0303 | 0.1497 |
| ONNX jit                   | 0.4406 | 0.5901 |
| ONNX session creation      | 0.0419 | 0.0577 |
| pytorch compile            | 2.7447 | 3.7696 |
| pytorch eager:             | 0.0717 | 0.3047 |
| pytorch inference compiled | 0.2242 | 0.2307 |
| pytorch onnx export        | nan    | nan    |

## mobilenet_v2

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.3843 | 0.3902 |
| ONNX eager                 | 1.9250 | 5.9817 |
| ONNX infer from session    | 0.0270 | 0.1373 |
| ONNX jit                   | 0.4511 | 0.5680 |
| ONNX session creation      | 0.0423 | 0.0438 |
| pytorch compile            | 2.4110 | nan    |
| pytorch eager:             | 0.0597 | 0.2804 |
| pytorch inference compiled | 0.1705 | nan    |
| pytorch onnx export        | nan    | nan    |

## resnet50

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.7309 | 0.6336 |
| ONNX eager                 | 2.6525 | 9.5537 |
| ONNX infer from session    | 0.1679 | 0.8601 |
| ONNX jit                   | 0.8636 | 1.5759 |
| ONNX session creation      | 0.0859 | 0.1088 |
| pytorch compile            | 2.8200 | nan    |
| pytorch eager:             | 0.2329 | 1.1624 |
| pytorch inference compiled | 0.1861 | nan    |
| pytorch onnx export        | nan    | nan    |

## shufflenet_v2_x1_0

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.7751 | 0.7106 |
| ONNX eager                 | 1.3031 | 2.5980 |
| ONNX infer from session    | 0.0110 | 0.0553 |
| ONNX jit                   | 0.7853 | 0.8325 |
| ONNX session creation      | 0.0628 | 0.0761 |
| pytorch compile            | 2.9187 | nan    |
| pytorch eager:             | 0.0225 | 0.1024 |
| pytorch inference compiled | 0.0466 | nan    |
| pytorch onnx export        | nan    | nan    |

## MNIST

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.1159 | 0.1052 |
| ONNX eager                 | 0.1863 | 0.1703 |
| ONNX infer from session    | 0.0001 | 0.0003 |
| ONNX jit                   | 0.1213 | 0.1115 |
| ONNX session creation      | 0.0080 | 0.0072 |
| pytorch compile            | 0.1680 | 0.2662 |
| pytorch eager:             | 0.0004 | 0.0009 |
| pytorch inference compiled | 0.0001 | 0.0003 |
| pytorch onnx export        | 0.3152 | 0.2716 |

## SqueezeNet

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.0771 | 0.0831 |
| ONNX eager                 | 0.2872 | 0.7830 |
| ONNX infer from session    | 0.0187 | 0.0930 |
| ONNX jit                   | 0.1020 | 0.1778 |
| ONNX session creation      | 0.0073 | 0.0225 |
| pytorch compile            | 0.5901 | 1.1605 |
| pytorch eager:             | 0.0267 | 0.1403 |
| pytorch inference compiled | 0.0206 | 0.0981 |
| pytorch onnx export        | 0.3995 | 0.3983 |

## SuperResolutionNet

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.0132 | 0.0126 |
| ONNX eager                 | 0.2888 | 1.4699 |
| ONNX infer from session    | 0.1315 | 0.6718 |
| ONNX jit                   | 0.1563 | 0.7429 |
| ONNX session creation      | 0.0099 | 0.0506 |
| pytorch compile            | 0.3007 | 1.5554 |
| pytorch eager:             | 0.1862 | 0.9849 |
| pytorch inference compiled | 0.1733 | 0.8643 |
| pytorch onnx export        | 0.2684 | 0.2678 |

## dcgan

| Batch size                 | 2      | 10     |
| -------------------------- | ------ | ------ |
| ONNX build                 | 0.0554 | 0.0550 |
| ONNX eager                 | 0.1109 | 0.1801 |
| ONNX infer from session    | 0.0063 | 0.0304 |
| ONNX jit                   | 0.0707 | 0.0927 |
| ONNX session creation      | 0.0074 | 0.0068 |
| pytorch compile            | 0.2404 | 0.4155 |
| pytorch eager:             | 0.0061 | 0.0279 |
| pytorch inference compiled | 0.0054 | 0.0273 |
| pytorch onnx export        | nan    | nan    |
