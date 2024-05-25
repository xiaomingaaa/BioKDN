# Learning to Denoise Biomedical Knowledge Graph for Robust Molecular Interaction Prediction

This is the code necessary to run experiments on the BioKDN algorithm described in the paper [Biomedical Knowledge Graph-enhanced Denoising Network]().

## Abstract
Molecular interaction prediction plays a crucial role in forecasting unknown interactions between molecules, such as drug-target interaction (DTI) and drug-drug interaction (DDI), which are essential in the field of drug discovery and therapeutics. Although previous prediction methods have yielded promising results by leveraging the rich semantics and topological structure of biomedical knowledge graphs (KGs), they have primarily focused on enhancing predictive performance without addressing the presence of inevitable noise and inconsistent semantics. This limitation has hindered the advancement of KG-based prediction methods. To address this limitation, we propose BioKDN (Biomedical Knowledge Graph Denoising Network) for robust molecular interaction prediction. BioKDN refines the reliable structure of local subgraphs by denoising noisy links in a learnable manner, providing a general module for extracting task-relevant interactions. To enhance the reliability of the refined structure, BioKDN maintains consistent and robust semantics by smoothing relations around the target interaction. By maximizing the mutual information between reliable structure and smoothed relations, BioKDN emphasizes informative semantics to enable precise predictions. Experimental results on real-world datasets show that BioKDN surpasses state-of-the-art models in DTI and DDI prediction tasks, confirming the effectiveness and robustness of BioKDN in denoising unreliable interactions within contaminated KGs.

## Requiremetns

All the required packages can be installed by running `pip install -r requirements.txt`.
```
dgl==1.1.2
lmdb==1.4.1
networkx==3.0
scikit-learn==0.22.1
torch==2.0.0
tqdm==4.61.2
```

## Experiments

### DTI prediction
To test BioKDN on the DTI task, run the following command. 
- `python main.py --task_type dti --dataset drugbank`


### DDI prediction
To test BioKDN on the DDI prediction task, run the following commands.
- `python main.py --task_type ddi --dataset drugbank`

### Reproduction
To reproduce the results of BioKDN and its variants, we will release the checkpoints and guidelines of them on [Google Driver](). 

<!--
## Acknowledgment
If you make use of this code or the SDN algorithm in your work, please cite the following paper:

	@article{ma,
	  title={Semantic-aware Denoising Network for Robust Inductive Knowledge Graph Completion},
	  author={},
	  journal={},
	  year={}
	}
 --!>
 

The code is implemented based on GraIL (https://github.com/kkteru/grail), in which we evaluate our model on the original inductive datasets proposed by the authors of GraIL. We thank you very much for their code sharing. We release a portion of the code (model files are excluded) because this paper is under review, and we will publish all the code and checkpoints once the article is accepted.
