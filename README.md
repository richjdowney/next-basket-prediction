# Generating "next basket" predictions using sequence models

### Project scope

This project contains a pipeline, orchestrated with Airflow, for generating predictions for a customers next basket
using sequence models.  

The initial version of the code utilizes an RNN model to generate probabilities that items will be found within the
customers next transaction.

### Data utilized

The data used for the pipeline was provided open source by Dunnhumby, a global leader in retail analytics. The dataset 
contained a sample of 117 weeks of ‘real’ customer data from a large grocery store constructed to replicate typical 
patterns found in in-store data to allow the development of algorithms in a (near) real-world environment. 

The dataset can be downloaded from the following location:

https://www.dunnhumby.com/careers/engineering/sourcefiles

The actual data utilized was from the “Let’s get sort of real” section, specifically the data from a randomly selected 
group of 5,000 customers.

### Infrastructure

The infrastructure utilized in the pipeline is shown in the diagram below:

![](Images/infrastructure.PNG)

The decision to utilize Spark was taken as retailer data is typically very large and this use-case requires each transaction to be 'scored' with a 'Shopping Mission'.  As the number of transactions typically runs into millions scaleability quickly becomes an issue with standard Python libraries.

PyCharm was utilized as the IDE and code was automatically deployed to an ec2 instance with Airflow installed with a Postgres RDS instance.  Data was stored in an s3 bucket, data processing and modelling is run with PySpark and SparkML.  

### Airflow Orchestration

In order to run Airflow it was installed on the same EC2 cluster where the code is deployed.  Steps to install Airflow using a Postgres database can be found [here](https://medium.com/@abraham.pabbathi/airflow-on-aws-ec2-instance-with-ubuntu-aff8d3206171)

The image below illustrates the orchestration of the tasks within Airflow:

![](Images/airflow_DAG.PNG)  

The DAG contains the following tasks:

**create_app_egg:**  Creates an egg file from the latest code  
**upload_app_to_s3:**  Uploads the application egg and Spark runner files containing the main functions to S3  
**create_job_flow:**  Creates an EMR cluster  
**branching:**  Determines if new models should be trained or if an existing model should be utilized to score a new transaction file  
**add_step_XXX:**  Adds Spark steps for staging data, pre-processing data, training and tuning LDA models, profiling and model scoring  
**watch_stage_XXX:**  Sensors for each staging step to determine when they are complete  
**remove_cluster:**  Terminates the cluster when all steps are completed  

### Model Details

In order to create 'Shopping Missions' LDA was utilized in a similar way to topic modelling from NLP.  Each transaction was considered to be a document and the items words within a document.  

The diagram below illustrates 2 examples of the types of 'Shopping Missions' that can be found utilizing this technique:  

![](Images/mission_examples.PNG)

LDA is particularly suited to this task because it is not influenced by the order in which the items are added to the basket in the same way that the order of the words in a document does not influence the topic to which the document is assigned.  
