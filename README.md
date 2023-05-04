# A Critical Look at Recent Trends in Compression of Channel State Information (ICASSP 2023)

This is a PyTorch compatible implementation of the transform coding scheme used in

```
@inproceedings{valtonen-ornhag-etal-icassp-2023,
  author       = {Marcus {Valtonen~{\"O}rnhag} and 
                  Stefan Adalbj{\"o}rnsson and 
                  P{\"u}ren G{\"u}ler and 
                  Mojtaba Mahdavi},
  title        = {A Critical Look at Recent Trends in Compression of Channel State Information},
  booktitle    = {To be presented at the {IEEE} International Conference on Acoustics, Speech and Signal Processing,
                  {ICASSP} 2023},
  year         = {2023},
}
```

Please cite the paper if you use the code in academic publications.

## Before using the code (important!)
Please note, that the core message from our paper is *not* to propose a state-of-the art method, but to
have an open discussion on the datasets and metrics used by the community. In particular, we propose to:

* Utilize new datasets, e.g. [DeepMIMO](https://deepmimo.net).
* Always quantize the data. Do not measure compression ratio in terms of length of the vector!
* Be mindful about using the NMSE metric for intermediate system-level simulations.
* Bring research from academia and industry closer, not further apart.

## Getting started
### Dataset
Our work is using the COST2100 dataset by [Wen et al.](https://ieeexplore.ieee.org/document/6393523),
which can be downloaded from one of the following locations:

 * [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing)
 * [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA)

Download the dataset and put it in a folder named `COST2100` in the root folder.

### Setting up an environment
You can use a virtual environment to run the code:
```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
We used Python 3.9.14 on UNIX, but the scripts should be compatible for Python versions >= 3.7.

## Replicating the results
### Running the transform coding scheme
You can now execute the main script
```
python main.py --data-dir COST2100 --scenario in -b 200 -j 0 --cr 4 --cpu --evaluate
```
To recreate the results from the paper run with `--datadir {in,out}` and `--cr {4,16,64}`.


| Scenario | Compression Ratio |  NMSE   |  rho  |
|:--------:|:-----------------:|:-------:|:-----:|
|  indoor  |        1/4        | -25.75* | 0.99  |
|  indoor  |       1/16        | -15.39  | 0.98  |
|  indoor  |       1/64        |  -8.73  | 0.94  |
| outdoor  |        1/4        | -17.67  | 0.97  |
| outdoor  |       1/16        |  -9.81  | 0.93  |
| outdoor  |       1/64        |  -4.61  | 0.77  |

*In the paper we reported 25.78, which is a typo.

**Note:** The code is *not* optimized for performance. By lowering the bit precision and increasing
the number of coefficients the performance could be further increased. This, however, is not the main
point of this paper.

### Quantization experiments
Use the `--bit-depth` flag, e.g.
```
python main.py [...] --cr 16 --bit-depth 4
```
which would result in a comparison where the corresponding deep-learning based method is sending a vector
reduced by 1/16 of its original length together and where each real and imaginary bit is allocated 4 bits.

### Comparison to other methods
In the quantization experiments, we used the open-source repositories from

 * Lu et al. (2020), [CRNet repo](https://github.com/Kylin9511/CRNet)
 * Cui et al. (2022), [TransNet repo](https://github.com/Treedy2020/TransNet)

Our code conform with the above-mentioned repositories, in order to utilize the same evaluation script.
To test the quantization impact of these methods one needs to modify the encoder output and decoder
input by passing them through the non-linear quantization methods available in `models/quantization`.