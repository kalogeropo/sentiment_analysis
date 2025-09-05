# Machine Learning Pipeline

This repository is part of the **Libeccio Programme**, a data-driven platform for tourism and regional development.  
The Machine Learning Pipeline provides the core infrastructure for experimenting with, evaluating, and deploying ML models on top of heterogeneous tourism-related data (e.g. reviews, mobility, climate, aviation, accommodation).

The pipeline’s main goals are:
- **Data Integration**: consume multilingual, multi-source tourism data streams collected within Libeccio (reviews, bookings, events, environmental signals).  
- **Experiment Management**: track parameters, metrics, and artifacts with [MLflow](https://mlflow.org/) for reproducibility and transparent comparison of models.  
- **Modular Workflows**: each task (e.g. sentiment analysis on reviews, demand forecasting, anomaly detection in tourism flows) is encapsulated as a reusable component that can be combined into larger workflows.  
- **Evaluation & Benchmarking**: standardized evaluators for classification, regression, and translation tasks (accuracy, F1, BLEU/chrF, RMSE, etc.), enabling fair comparison across domains and models.  
- **Deployment**: package best-performing models as MLflow `pyfunc` models, ready for integration into the Libeccio DMSS dashboards and APIs.

### Example Use Case: Sentiment Analysis
Tourism reviews in multiple languages are processed through the pipeline:
1. **Language Detection** → identify the review’s source language.  
2. **Translation** → translate to English using the best-performing model for that language (benchmarked in this pipeline).  
3. **Sentiment Classification** → classify as positive, neutral, or negative.  
4. **Logging & Selection** → log results to MLflow; select the optimal detector/translator/classifier path per language.  

This ensures that multilingual feedback from visitors is consistently analyzed and comparable, supporting Libeccio’s broader goal of providing **decision-support insights for tourism stakeholders**.

---.


## Sentiment Analysis with Multilingual Translation

This project explores **sentiment analysis on multilingual reviews**, focusing on:
1. **Language Detection** – choosing the most reliable detector for noisy text.  
2. **Machine Translation** – translating reviews to English with different methods (currently Opus-MT, more can be added).  
3. **Downstream Sentiment** – (planned) evaluating how translation quality affects English sentiment classification.  

The **end goal** is to identify the **best translation pipeline per source language** by maximizing downstream **sentiment accuracy**.

## The rest of machine learning things we will do here

## Getting started

1. Check remotes

```
git remote -v
```

Here should be two repositories. The github is public and will be used as a referance repository for research purposes in the future.
Therefore, for the Libeccio Project any changes should happen in the **GITLAB** dev branch.
**ITS IS IMPORTANT** to use best practises and any file that contains **sensitive information** should be added to the **.gitignore**.

2. Create or switch to a branch

```
# update local refs
git fetch gitlab

# create a new branch from GitLab main
git checkout -b my-feature gitlab/main
```

3. Commit your changes

```
git add <files>
git commit -m "Meaningful commit message"
```
4. Push changes to GitLab
```
# first push sets upstream
git push -u gitlab my-feature

# for later commits
git push
```
5. Update with changes from remote repository (It is not needed when working with ssh in the Libeccio ML computer)
```
git pull 
```

Most of this work can be done via the IDE either VScode or Pycharm.
Merge requests after milestones can be done via the gitlab user interface.



## Installation

```
conda env create -f environment.yml
conda activate sentiment-analysis-env
```

## Usage

1. Sentiment Analysis with Multilingual Translation will implement mlflow for development and deployment. In later time we will offer examples.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
