# Romantic Chaos: A study of the emergence of unpredictability in relationships
As part of the course "Complex System Simulation" - University of Amsterdam 2024
Used to support the final presentation of the project. This includes creating grpahs, visualizations and animations, as well as to get a deeper understanding of the model and the dynamics of the model. 

### License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains Python scripts to create graphs for our final presentation. TO be able to run the scripts see the requirements below. By clicking on the run option in the .py file or running each cell in the .ipynb file, the graphs will be created. 

## File Descriptions

### `blabla.py`
This file includes the core methods used in this project. It contains the implementation of the basic models. 

### `antithetic.py`
`blabla.py` implements 


## Usage
These scripts were run with Python 3.11.0 on MacOS Ventura. 


### Requirements:
* matplotlib==3.7.1 
* scipy==1.10.1
* networkx==3.1
* ipywidgets==8.0.4
* numpy==1.24.3
* pickle (in standard package)
* mpl_toolkits (in standard package)
* plotly==5.9.0
* pandas==1.5.3
* multiprocessing
* tqdm==4.65.0
* ipython==8.12.0
* time (in standard package)
* os (in standard package)
* unittest (in standard package)
* imageio==2.31.1
* PIL==10.2.0 (also called Pillow)
* matplotlib-inline==0.1.6



### References:
The following material inspired and provided the equations and initial parameter values for our model:
* Rinaldi, S., Rossa, F. D., Dercole, F., Gragnani, A., & Landi, P. (2015). Modeling Love Dynamics: Vol. Volume 89. WORLD SCIENTIFIC. https://doi.org/10.1142/9656
* ERBAŞ, Kadir Can. “Modeling Love with 4D Dynamical System.” Chaos Theory and Applications, vol. 4, no. 3, Akif Akgul, 30 Nov. 2022, pp. 135–143. doi:10.51537/chaos.1131966.​


## Contact
sandor.battaglini-fischer@student.uva.nl
victor.piaskowski@student.uva.nl

---

Developed by Sándor Battaglini-Fischer, Yehui He, Thomas Norton, Victor Piaskowski

# complex-system-simulation

Parameter explaination:
alpha1    =   Forgetting coefficient 1 (decay rate of love of individual 1 in absence of partner)
alpha2    =   Forgetting coefficient 2
beta1     =   Reactiveness to love of 2 on 1 (influence of the partner's love on an individual's feelings)
beta2     =   Reactiveness to love of 1 on 2
gamma1    =   Reactiveness to appeal of 2 on 1 (influence of the partner's appeal on an individual's feelings)
gamma2    =   Reactiveness to appeal of 1 on 2
bA1       =   Bias coefficient of individual 1 (how much individual 1 is biased towards their partner, > 0 for synergic, 0 for unbiased, < 0 for platonic)
bA2       =   Bias coefficient of individual 2
A1        =   Appeal of individual 1 (how much individual 1 is appealing to their partner)
A2        =   Appeal of individual 2
k1        =   Insecurity of individual 1 (Peak of reaction function of 1 on 2, high k1 means they are annoyed by their partner's love earlier)
k2        =   Insecurity of individual 2
n1        =   Shape of reaction function of 1 on 2 (nonlinearity of reaction function of 1 on 2, sensitivity of the individuals' feelings to changes in their partner's feelings)
n2        =   Shape of reaction function of 2 on 1
m1        =   Shape of bias function of 1 (nonlinearity of bias function of 1, sensitivity of how the own feelings influence their perception of their partner's appeal)
m2        =   Shape of bias function of 2
sigma1    =   Saddle quantity of 1 (Trace of Jabobian of 1, threshold of when own feelings influence their perception of their partner's appeal. > 0 for stable, < 0 for unstable)
sigma2    =   Saddle quantity of 2


For the love and move_multiple files, the parameters and its meanings are here are as follows:
axx = Forgetting coefficient of the intimacy of Xena to Yorgo.
axy = If Yorgo’s intimacy increases, Xena’s will decrease, and if it decreases, it will increase. If Xena’s partner shows closeness/interest to her, Xena gradually loses her sense of intimacy.
bxx = If Xena’s passion increases, her sense of intimacy increases, and if it decreases, it decreases. She is intimate with someone she is passionate about. She might just want to fall in love.
bxy = As Yorgo’s passion for Xena increase, Xena’s closeness to Yorgo decreases. When she realizes that Yorgo is not in love, Xena increases her intimacy. Maybe she doesn’t want someone in love with her.
cxx = Her passion increase when Xena feels close. Men with whom she does not feel close are not attractive, but men with whom she feels sincere can be attractive.
cxy = Intimate men are very attractive to Xena. Her passion for men who do not behave closely is significantly reduced.
dxx = Forgetting coefficient of the passion of Xena for Yorgo.
dxy = As Yorgo’s passion grows, so does Xena’s. A man who acts romantic may attract her.
fxy = Xena’s impression of intimacy or friendship with Yorgo. Xena finds Yorgo intimate and friendly. She enjoys being friends and spending time with him.
gxy = Xena’s impression of glamorousness or attractiveness about Yorgo. Xena does not find Yorgo romantically or sexually attractive.
ayy = Forgetting coefficient of the intimacy of Yorgo to Xena.
ayx = If Yorgo’s intimacy increases, Xena’s will decrease, if it decreases, it will increase. If Yorgo’s partner shows intimacy/interest to him, Yorgo increases his sense of intimacy.
byy = If Yorgo’s passion increases, his sense of intimacy decreases, and if it decreases, it increases. He is intimate with someone he is not passionate about. He might just want not to fall in love.
byx = As Xena’s passion for Yorgo increases, Yorgo’s intimacy with Xena increases. When he realizes that Xena is not in love, Yorgo decreases his intimacy. Maybe he wants someone in love with her.
cyy = His passion decreases when Yorgo feels close. Women with whom he does not feel close are attractive, but women with whom he feels intimacy are not attractive.
cyx = Intimate women are not attractive to Yorgo. His passion for women who are close to him weakens a little.
dyy = Forgetting coefficient of the passion of Yorgo to Xena.
dyx = As Xena’s passion increases, Yorgo’s decreases. A romantic woman does not attract him.
fyx = Yorgo’s impression of intimacy or friendship with Xena. Yorgo found Xena neither sympathetic nor antisympathetic.
gyx = Yorgo’s impression of glamorousness or attractiveness about Xena. Yorgo finds Xena attractive. He desires her romantically and sexually.
