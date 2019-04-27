## neura

###### yet another comparison of neural net archs for image enhancement task

![](results/src.jpeg)
_source image_

![](results/interpolation_nearest.jpeg)
_interpolation nearest neighbor [see](source/models/interpolation_nearest.py)_

![](results/interpolation_bilenear.jpeg)
_interpolation bilenear [see](source/models/interpolation_bilenear.py)_

![](results/interpolation_bicubic.jpeg)
_interpolation bicubic [see](source/models/interpolation_bicubic.py)_

![](results/interpolation_lancoz.jpeg)
_interpolation lancoz [see](source/models/interpolation_lancoz.py)_

![](results/convolution_scaled.jpeg)
_prescaled convolution net [see](source/models/convolution_scaled.py)_

![](results/convolution.jpeg)
_upsampling convolution net [see](source/models/convolution.py)_

![](results/convolution_avg.jpeg)
_upsampling convolution net with avg layers [see](source/models/convolution_avg.py)_

![](results/convolution_denoise.jpeg)
_upsampling convolution net with denoise layers [see](source/models/convolution_denoise.py)_

![](results/convolution_rec.jpeg)
_upsampling convolution net with rec layers [see](source/models/convolution_rec.py)_

neura was built with:

- python3
- keras
- pyplot

#### LICENSE

neura is [MIT licensed](LICENSE)
