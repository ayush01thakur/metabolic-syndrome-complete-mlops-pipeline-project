# METABOLIC SYNDROME MLOPS COMPLETE PIPELINE PROJECT (INCLUDING WEB-APP)

This project predicts the likelihood of metabolic syndrome based on medical test reports. It uses machine learning techniques to analyze various health parameters and determine if an individual is at risk of metabolic syndrome.

![image](https://github.com/ayush01thakur/metabolic-syndrome-complete-mlops-pipeline-project/assets/124871122/98168461-70e7-4bf2-9cde-1169742e2cde)

## Installation

1. Clone this repository:
2. Either install the requirements or just run the setup.py, it will install the requirements as well
   
   **command to install just requirements**
   ```bash
   pip install -r requirements.txt
   ```

   **or run the setup.py file**
   ```bash
   python setup.py
   ```
3. Need to have HTML, CSS, JS pre-installed to run the web application.

## About features used and dataset
This dataset contains information on individuals with metabolic syndrome, a complex medical condition associated with a cluster of risk factors for cardiovascular diseases and type 2 diabetes. The data includes demographic, clinical, and laboratory measurements and the presence or absence of metabolic syndrome.

Column Descriptors:
- seqn: Sequential identification number.
- Age: Age of the individual.
- Sex: Gender of the individual (e.g., Male, Female).
- Marital: Marital status of the individual.
- Income: Income level or income-related information.
- Race: Ethnic or racial background of the individual.
- WaistCirc: Waist circumference measurement.
- BMI: Body Mass Index, a measure of body composition.
- Albuminuria: Measurement related to albumin in urine.
- UrAlbCr: Urinary albumin-to-creatinine ratio.
- UricAcid: Uric acid levels in the blood.
- BloodGlucose: Blood glucose levels, an indicator of diabetes risk.
- HDL: High-Density Lipoprotein cholesterol levels (the "good" cholesterol).
- Triglycerides: Triglyceride levels in the blood.
- MetabolicSyndrome: Binary variable indicating the presence (1) or absence (0) of metabolic syndrome.

We will drop some features during the feature selection process, have a look at the EDA file in the raw_data directory to understand more about how feature selection is done there.

## How to run the project.
1. First install all the requirements.
2. run the `app.py` file inside the website directory in cmd or any terminal you running in you environment.

   ```bash
   python website/app.py
   ```
3. The flask server will start running on port 5000. you can visit the site on the localhost address: `http://127.0.0.1:5000`
   If you are still not sure check the logs folder which will be created once you run the app.py (1). There you can find the latest log file in which you have the localhost web address to visit the site.
   ![image](https://github.com/ayush01thakur/metabolic-syndrome-complete-mlops-pipeline-project/assets/124871122/715ae280-fb9e-4264-9a85-beb5a9643c99)

   These log files will be created whenever you run the app.py or any pipeline files like `run_training_pipeline.py` or `run_prediction_pipeline.py`. These will help you to navigate the workings of the project and will also help in identifying the errors that occur.

4. Fill in the details for the asked fields, and hit the submit button. You can hover over the `i` icon for the details of the asked fields.
5. You will see the results below the submit button.



## Pipeline and Component Features

   `model_components` : This folder contains all the major component files or dependencies for the traning of the model.
      - Data ingestion: This file ingest the data from various sources (here from csv file stored inside the `raw_data` directory).
      - Preprocessing and Transformation: This file  preprocess and transform data according the analysis done in EDA `eda.ipynb` file  in `raw_data` directory. This will produce a `preprocessing.pkl` file (stored inside `artifacts` folder) which will help the preprocessing of the input data while prediction.
      - ML Model Development: Before model development this file creates several machine learning models and compares their accuracies. It select the model with the highest accuracy and then stores the `model.pkl` file inside `artifacts` folder, which will later helps in prediction. 

   `pipelines` : This stores the major pipeline which is training and prediction pipeline, later will add deployment pipeline as well.
      - training pipeline: this sets up the base for the prediction pipeline as the prediciton pipeline cannot run without the  `model.pkl` and `preprocessing.pkl` files which is produced during this pipeline's execution.
      - prediction pipeline: this will predic the results and send back to the `app.py` file to disply on the website.

  
