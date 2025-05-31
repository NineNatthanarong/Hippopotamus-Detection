# Project Setup and Execution

This guide outlines the steps to set up and run the project, including downloading the dataset, training the model, and testing its performance.

## 1. Download the Dataset

Execute the following command in your terminal to download and extract the dataset:

```bash
curl -L "https://app.roboflow.com/ds/rWSEyWjNTf?key=vdc6Vyh3kF" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

This command performs the following actions:

-   `curl -L "https://app.roboflow.com/ds/rWSEyWjNTf?key=vdc6Vyh3kF" > roboflow.zip`: Downloads the dataset from the specified Roboflow link and saves it as `roboflow.zip`.
-   `unzip roboflow.zip`: Extracts the contents of the `roboflow.zip` archive.
-   `rm roboflow.zip`: Removes the `roboflow.zip` file after extraction.

## 2. Train the Model

To train the model, run the `main.py` script:

```bash
python main.py
```

This script will initiate the training process using the downloaded dataset. Ensure that all necessary dependencies are installed before running this command.

## 3. Test the Model

After training, you can test the model's performance by running the `test.py` script:

```bash
python test.py
```

This script will evaluate the trained model on the test dataset and output the performance metrics.