# Machine Learning 

Directory withing the laser runner removal project for preparing data and training ML models 

## Environment setup

1.  Make sure you are in the ml_model repo 

1.  Create and source into a venv

        $ python3 -m venv venv
        $ source venv/bin/activate

1.  Install necessary requirements 

        $ pip install -r requirements.txt

## DVC setup 

DVC is a project that abstracts cloud storage of large data for machine learning. It allows a directory structure to be push and pull from a s3 bucket, so all developers can expect everyone to have the same folder structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines. 

1.  Setup a dvc repo, 

        $ dvc init 

1.  Setup aws s3 permission this will require an aws account key and secret ID. NOTE: We should move to a more secure method over time. 

        $ dvc config 

1.  Add the laser-runner-removal-dvc s3 bucket as a remote repository and pull data from it

        $ dvc remote add segmentation_data s3://laser-runner-removal-dvc
        $ dvc pull -r segmentation_data 






