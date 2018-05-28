# LogicalRegressionDemo

## Vereisten:

De demo is beschikbaar in 2 versies. Beide demo versies zijn gemaakt en getest op een 64-bits Windows systeem.

[Versie 1](https://github.com/HanNotSMachineLearning/LogisticRegressionDemo/blob/master/LogisticRegressionDemo.py) vereisten:

* pip 10.0.1
* Python 3.6.5

[Versie 2](https://github.com/HanNotSMachineLearning/LogisticRegressionDemo/blob/master/logistic_regression.ipynb) vereisten:

* pip 10.0.1
* Python 3.6.5
* Jupiter notebook 5.5.0

## Installatie

> versie 1 is te runnen door naar het hoofd bestandsmap (LogisticRegressionDemo) de commando **python LogisticRegressionDemo.py** uit te voeren.
> versie 2 is te runnen door naar het hoofd bestandsmap (LogisticRegressionDemo) de commando **jupyter notebook logistic_regression.ipynb** uit voeren. Daarna kan de demo gerunt worden door te .... selecteren.

Running the demo: ![Run demo](https://github.com/HanNotSMachineLearning/LogisticRegressionDemo/blob/master/Re_Run.PNG "Re-Run")

Om de programmacode te draaien heb je de volgende Python-modulen nodig:

* pandas
* seaborn
* statsmodels
* patsy

Door het commando **pip install -r requirements.txt --user** uit te voeren in een opdrachtvenster worden alle modules in één keer gedownload.

## Data

bank-names.txt: relevante informatie over de banking.csv

banking.csv: dataset die gebruikt wordt voor de demo.

* [Deze is afkomstig van (raw bestand opslaan als en in hoofdmap te zetten voor gebruik)](https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv)
* [Dataset is ook te vinden in de UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## Demo code

> [Versie 1](https://github.com/HanNotSMachineLearning/LogisticRegressionDemo/blob/master/LogisticRegressionDemo.py): De demo(LogisticRegressionDemo.py) is gescreven in python en is een eenvoudige demo zonder veel documentatie.

> [versie 2](https://github.com/HanNotSMachineLearning/LogisticRegressionDemo/blob/master/logistic_regression.ipynb): De demo(logistic_regression.ipynb) is gemaakt met behulp van jupyter notebook en python. De demo bevat de volledige documentatie en details zoals grafieken.

## Resultaat van de applicatie

Het resultaat van de applicatie is dat de structuur van logistic regression algoritme wordt getoond. Hierbij wordt er voorspeld of een klant een termijn deposito zal overschijven of niet aan de hand van bestaande data.
