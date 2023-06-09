# Approximate Text Explanation (ATE)
Transformation of TensorFlow text classification models into local interpretable models to explain the base model decisions via effects.

Related work and base for this idea:
<ul>
<li><a href="https://arxiv.org/abs/1602.04938">Paper: "Why Should I Trust You?": Explaining the Predictions of Any Classifier by Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin</a></li>
<li><a href="https://github.com/marcotcr/lime">GitHub: marcotcr/lime</a></li>
</ul>

## Approximate Local Text Explanation (ALTE)
"Approximate Local Text Explanation" is based on the LIME method. The goal of the approach is the derivation of effects or influences of the input components (textual data) on the respective output (classification label). This is a local explanation procedure in which a single input data point is analyzed. The components of this input data point (token) are activated or deactivated by permutations of a binary vector of the same size as the number of components of the input data point. All permutations are classified by the original classification model and stored in a meta dataset. This meta dataset is then used to train a linear classification model, thus linearly approximating the original classification function. Since the computation of all permutations of the components of the input data point is very computationally expensive, the permutation upper bound, the permutation repetitions and the epochs of the linear model can be defined via configuration parameters. In addition, the permutation process is iteratively repeated and the permutations are randomized, which also makes it possible to refine the linear model in the long run.

## Approximate Global Text Explanation (AGTE)
"Approximate Global Text Explanation" is also based on the LIME method. The goal of the approach is to infer effects or influences of the input components (textual data) of all input data points in the data set on the respective output (classification label). This is a global explanation procedure in which the effects of all input data points are analyzed. AGTE uses the approach of ALTE for this and extends it with an N-fold execution. Accordingly, several linear models are trained which represent the original classification function at different points. The more data points exist and the more computational capacity is available (i.e. the more permutations can be calculated and classified), the better the approximation of the original model. In addition, a pipeline is implemented, with the help of which the effect representation (token and its effect on certain classes) can be converted into a decision rule set (DecisionTree).

## How to use?
The project contains two Jupyter notebooks in the tutorials folder, which are used to illustrate the concepts of Approximate Local Text Explanation (ALTE) and Approximate Global Text Explanation (AGTE) using the IMDb Movie Review dataset.
- <a href="https://github.com/tjarkpr/ate/blob/main/tutorials/alte.ipynb">Tutorial: Approximate Local Text Explanation (ALTE) - IMDb movie review sentiment explanation</a>
- <a href="https://github.com/tjarkpr/ate/blob/main/tutorials/agte.ipynb">Tutorial: Approximate Global Text Explanation (AGTE) - IMDb movie review sentiment rules explanation</a>