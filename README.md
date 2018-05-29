# DEC clustering in pyTorch
This is an implementation of Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016 https://arxiv.org/pdf/1511.06335.pdf
## Pre-requsit
* pyTorch 0.3+ with CUDA environment
* torchVision
* scikit-learn
## Usage
python DEC.py
## Results
This code can reach around 87% accuracy on mnist test dataset http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz (trained on mnist training sets)
## References
* The code references https://github.com/XifengGuo/DEC-keras and reuse some of the code (thanks for the good work :)
* To keep in mind of the bigger dataset, miniBatchKMeans is used to get cluster center.
* Right now only mnist dataset is tested, it should be easier to add more dataset through torchVision DataLoader.
