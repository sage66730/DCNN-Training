# DCNN-Training
This repository is the official implementation of [ON TRAINING DERIVATIVE-CONSTRAINED NEURAL NETWORKS](https://arxiv.org/abs/2030.12345). 

![method_001](https://github.com/lbai-lab/integrated-activation-functions/assets/28528165/8da5d65b-42aa-478b-b9a6-a0f5b7ae27b8)

We refer to the setting where the (partial) derivatives of a neural network’s (NN’s)
predictions with respect to its inputs are used as additional training signal as a
derivative-constrained (DC) NN. This situation is common in physics-informed
settings in the natural sciences. We propose an integrated RELU (IReLU) acti-
vation function to improve training of DC NNs. We also investigate denormal-
ization and label rescaling to help stabilize DC training. We evaluate our meth-
ods on physics-informed settings including quantum chemistry and Scientific Ma-
chine Learning (SciML) tasks. We demonstrate that existing architectures with
IReLU activations combined with denormalization/label rescaling better incorpo-
rate training signal provided by derivative constraints

Benchmarked models: 
- CGCNN [[`arXiv`](https://arxiv.org/abs/1710.10324)] 
- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)] 
- ForceNet [[`arXiv`](https://arxiv.org/abs/2103.01436)] 
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] 
- GemNet-dT [[`arXiv`](https://arxiv.org/abs/2106.08903)] 

## Requirements
To install requirements with conda:
```setup
conda env create -f environment.yml
```

## Data
- Download the dataset for chemistry experimeants manually from [MD17](http://www.sgdml.org/#datasets) or run:
```download
cd ./datasets/Chemistry/MD17 
sh download_datas.sh 
```
A default random split of 80% train, 10% valid and 10% test will be generated.  

- Download the dataset for PINN experimeants manually from [PDEBench](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) or run:
```download
cd ./datasets/PDEBench/
sh download_datas.sh 
```
Datas of PINN are splitted while runing.

## Training
- To train the chemistry model(s) in the paper, run below:
```train
cd ./scirpts/Chemistry 
python train.py --root <path_to_data> -M <model_name> -m <molecule_abbreviation> -A <activation_function> --lr2 <lr_last_layer>
```

- To train the PINN model(s) from PDEBench in the paper, run below:
```train
cd ./scirpts/PDEBench 
python train.py --root <path_to_data> -M <scenario_name> -A <activation_function> --lr2 <lr_last_layer>
```

## Evaluation
In our training script, the evalueation would be conducted and recorded to Wandb right after training is finished.  
- To manually evaluate a trained chemistry model, run:
```eval
cd ./scirpts/Chemistry
python eval.py --root <path_to_data> --model_path <path_to_model_dir> --epoch <select_trained_epoch>
```

## Pre-trained Models
You can download pretrained models here:
- To be done

## Contributing
- To be done

