# DCST
DCST: Sparse Dictionary Coding Based Compression of Multivariate Time Series Using Spatiotemporal Correlation

- **Overview**: DCST is a new Multivariate time series compression, which based on sparse dictionary coding, using spatiotemporal correlation
- **Compression steps**:
  1、wavelet denosing
  2、normalization
  3、dividing into segments
  4、calculating correlation
  5、dictionary learning
  6、dictionary sparse Coding
  7、spatiotemporal compression


## Prerequisites
- Ubuntu 18 and 20 (including the same distribution under WSL) or Mac OS.
- Clone this repo
---

## Build
- To install  all the dependencies, run the following installations script:
```bash
    $ sh install.sh
```


## Run
- Directly run dcst.py

### Results
All the results including the compressed data, runtime, accuracy error, and the compression ratios will be add to 'results/{dataset_name}.txt' file. The results of the baseline TRISTAn and CORAD weill be also added.
