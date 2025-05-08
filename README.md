# DReKGNN: Drug repositioning based on expert knowledge augmented graph neural network
This is our tensorflow implementation of DReKGNN for drug repositioning.

 
## Environment Requirement
- tensorflow-gpu == 2.6.0
- keras == 2.6.0
- scikit-learn == 0.22.2
- torch == 1.13.0+cu117
- huggingface-hub == 0.27.0
- accelerate == 0.20.1
- beautifulsoup4 == 4.12.3
- bitsandbytes == 0.44.1



## Usage

Please set the mode parameter to "cv", and "analysis" in main.py to reproduce 10-fold cross-validation results, and discovering candidates for new diseases results reported in our paper, respectively.



A quick start example of 10-fold cross-validation is given by:

```shell
$ python main.py --dataset Fdataset --mode cv
```

An example of discovering candidates for new diseases is as follows:
```shell
$ python main.py --dataset Cdataset --mode analysis
```

If user wants to re-obtain the LLM node embedding derived from expert knowledge:
```shell
$ python emb_generate.py --dataset Fdataset
or 
$ python emb_generate.py --dataset Cdataset
```
