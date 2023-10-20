# Machine Learning 

Directory withing the laser runner removal project for preparing data and training ML models 

## Environment setup

1.  Install CUDA, this is a package that allows for model training on GPU's. This is all from https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

        $wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        $sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        $wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        $sudo apt-get update
        $sudo apt-get -y install cuda

1.  Make sure you are in the ml_model repo 

1.  Create and source into a venv

        $ python3 -m venv venv
        $ source venv/bin/activate

1.  Install a specific version of pytorch to match cuda versions and detect GPUs

        $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    NOTE: If you have any existing installs of touch torchvision or torchaudio in the venv, this will cause errors and they should be uninstalled 

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

## Update Ultralytics directory files 

The ultralytics repo uses a config file to read in some directory locations, changing those to point to the ml_models folder makes things a bit cleaner and easier. 

1. Update the directory paths, use `/home/{user_name}/` instead of `~`. 

        $ nano ~/.config/Ultralytics/settings.yml

        $ datasets_dir: ~/ros_ws/src/laser_runner_removal/ml_model/    
        $ weights_dir: ~/ros_ws/src/laser_runner_removal/ml_model/ultralytics/weights
        $ runs_dir: ~/ros_ws/src/laser_runner_removal/ml_model/ultralytics/runs


##  Train the model 

1.  The local train script uses the ultalytics repo to train a segmentation model on the runner data 

        $ python local_train.py 




