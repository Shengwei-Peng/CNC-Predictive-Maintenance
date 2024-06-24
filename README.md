# CNC-Predictive-Maintenance

This repository contains the code and data for a predictive maintenance system for CNC (Computer Numerical Control) machines. The system aims to predict potential failures and maintenance needs to minimize downtime and enhance operational efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Predictive maintenance involves using data analysis tools and techniques to detect anomalies in your operations and possible defects in equipment and processes so you can fix them before they result in failure. This project focuses on predictive maintenance for CNC machines, leveraging machine learning algorithms to predict failures based on historical data.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Shengwei0516/CNC-Predictive-Maintenance.git
    cd CNC-Predictive-Maintenance
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data

The dataset used in this project consists of historical data from CNC machines, including sensor readings, operational parameters, and maintenance logs. Below is a description of the main columns in the dataset:

<details>
<summary>Click to expand</summary>

| Column           | Description                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------|
| **time**         | Timestamp indicating when the data was recorded.                                                |
| **V_avg_machine**| The average voltage supplied to the CNC machine.                                                |
| **I_avg_machine**| The average current drawn by the CNC machine.                                                   |
| **kW_machine**   | Active power consumption of the CNC machine in kilowatts.                                       |
| **kvar_machine** | Reactive power consumption of the CNC machine in kilovars.                                      |
| **kVA_machine**  | Apparent power consumption of the CNC machine in kilovolt-amperes.                              |
| **PF_machine**   | Power factor of the CNC machine, representing the efficiency of power usage.                    |
| **kWh_machine**  | Total energy consumption of the CNC machine in kilowatt-hours.                                  |
| **kvarh_machine**| Total reactive energy consumption of the CNC machine in kilovar-hours.                          |
| **kVAh_machine** | Total apparent energy consumption of the CNC machine in kilovolt-ampere-hours.                  |
| **V_avg_spindle**| The average voltage supplied to the spindle.                                                    |
| **I_avg_spindle**| The average current drawn by the spindle.                                                       |
| **kW_spindle**   | Active power consumption of the spindle in kilowatts.                                           |
| **kvar_spindle** | Reactive power consumption of the spindle in kilovars.                                          |
| **kVA_spindle**  | Apparent power consumption of the spindle in kilovolt-amperes.                                  |
| **PF_spindle**   | Power factor of the spindle, indicating the efficiency of power usage.                          |
| **kWh_spindle**  | Total energy consumption of the spindle in kilowatt-hours.                                      |
| **kvarh_spindle**| Total reactive energy consumption of the spindle in kilovar-hours.                              |
| **kVAh_spindle** | Total apparent energy consumption of the spindle in kilovolt-ampere-hours.                      |
| **RPM**          | Rotational speed of the spindle in revolutions per minute.                                      |
| **Anomaly**      | Boolean indicator signifying whether the data point is considered an anomaly (`True`/`False`).  |

</details>

### Example Data

<details>
<summary>Click to expand</summary>

| time                | V_avg_machine  | I_avg_machine | kW_machine | kvar_machine | kVA_machine | PF_machine | kWh_machine | kvarh_machine | kVAh_machine | V_avg_spindle | I_avg_spindle | kW_spindle | kvar_spindle | kVA_spindle | PF_spindle | kWh_spindle | kvarh_spindle | kVAh_spindle | RPM  | Anomaly |
|---------------------|----------------|---------------|------------|--------------|-------------|------------|-------------|---------------|--------------|---------------|---------------|------------|--------------|-------------|------------|-------------|---------------|--------------|------|---------|
| 2023-05-04 21:06:25 | 222.5558624268 | 3.978289604   | 1.184034824| 0.9684904218 | 1.529677153 | 0.774042308| 1.816337466 | 1.503314137   | 2.358232737  | 360.457244873 | 5.533977985   | 0.152448997| 2.475371122  | 2.480061054 | 0.061469857| 0.164464176 | 1.580153108   | 1.598300338  | 2500 | False   |
| 2023-05-04 21:06:26 | 222.5959472656 | 3.980161905   | 1.184236288| 0.9688802361 | 1.530079126 | 0.773969958| 1.816667059 | 1.503583074   | 2.358657598  | 360.457244873 | 5.545718193   | 0.152648449| 2.477063417  | 2.481761694 | 0.061508474| 0.164548889 | 1.581528783   | 1.599682927  | 2500 | False   |
| 2023-05-04 21:06:27 | 222.6310272217 | 3.984233379   | 1.184538126| 0.9686712027 | 1.530180931 | 0.774116392| 1.816996574 | 1.503852010   | 2.359082460  | 360.457244873 | 5.550326347   | 0.152649045| 2.478666782  | 2.48336339  | 0.061468504| 0.164590508 | 1.582217336   | 1.600376129  | 2500 | False   |

</details>

## Usage

To utilize the predictive maintenance system, follow the steps below to run the main Jupyter Notebook (`main.ipynb`), which orchestrates the entire workflow from data preprocessing to model evaluation and visualization.

1. Ensure that you have followed the installation instructions to set up your environment.

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook main.ipynb
    ```

3. The `main.ipynb` notebook contains the code to preprocess data, train the model, evaluate its performance, and visualize the results. The main steps involved are:

    - **Initialize**: This involves setting up the necessary parameters.
    - **Preprocess**: This step involves cleaning and preparing the data for model training.
    - **Train**: The selected machine learning model is trained using the preprocessed data.
    - **Evaluate**: The trained model is evaluated to determine its accuracy and effectiveness in predicting failures.
    - **Visualize**: The results of the model's predictions are visualized.

4. **Models:**
    The system supports several machine learning models for predictive maintenance. You can specify the desired model using the `args.model` parameter. The available models are:

    - `NB`: Gaussian Naive Bayes
    - `KNN`: K-Nearest Neighbors
    - `DT`: Decision Tree
    - `RF`: Random Forest
    - `SVM`: Support Vector Machine
    - `MLP`: Multi-Layer Perceptron

5. **Arguments:**
    The arguments for running the system are defined as follows:
    <details>
    <summary>Click to expand</summary>

    | Argument       | Type   | Description                                                                   | Example             |
    |----------------|--------|-------------------------------------------------------------------------------|---------------------|
    | `file_path`    | str    | Path to the dataset file.                                                     | `"./data/data.csv"` |
    | `model`        | str    | The machine learning model to be used for prediction.                         | `"NB"`              |
    | `over_sampling`| bool   | Whether to apply over-sampling to balance the dataset.                        | `False`             |
    | `future_steps` | int    | Number of future steps to predict.                                            | `1`                 |
    | `window_size`  | int    | Size of the sliding window for feature extraction.                            | `3 `                |
    | `test_size`    | int    | Number of samples to include in the test split.                               | `200`               |
    | `seed`         | int    | Random seed for reproducibility.                                              | `0`                 |
    | `threshold`    | float  | Threshold value for converting the predicted probabilities into class labels. | `0.5`               |
    </details>

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw
