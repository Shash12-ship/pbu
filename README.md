# Partially Blinded Unlearning(PBU)

This repo is official implementation of the [Partially Blinded Unleanring(PBU): Class Unlearning for Deep Networks a Bayesian Perspective paper.](https://arxiv.org/abs/2403.16246)


------------------------------------
In order to adhere to regulatory standards governing individual data privacy and safety, machine learning models must systematically eliminate information derived from specific subsets of a user's training data that can no longer be utilized. The emerging discipline of Machine Unlearning has arisen as a pivotal area of research, facilitating the process of selectively discarding information designated to specific sets or classes of data from a pre-trained model, thereby eliminating the necessity for extensive retraining from scratch. The principal aim of this study is to formulate a methodology tailored for the purposeful elimination of information linked to a specific class of data from a pre-trained classification network. This intentional removal is crafted to degrade the model's performance specifically concerning the unlearned data class while concurrently minimizing any detrimental impacts on the model's performance in other classes.

![mechanism](/home/ece/Subhodip/Unlearning/pbu/PBU.png)

Partially Blinded Unlearning (PBU) Method: Given user-identified samples to be unlearned ($\mathcal{S}_n$), our unlearning method employs a two component perturbation technique, indicated by a loss function comprising three terms: the first term(shown in the bottom half) represents the perturbation in the output space, aiming to minimize the log-likelihood associated with the unlearned class while the last two terms correspond to perturbations in the parameter space (shown in the upper half), including the Mahalanobis Distance with respect to the Fisher Information matrix and the $l_2$ distance.
