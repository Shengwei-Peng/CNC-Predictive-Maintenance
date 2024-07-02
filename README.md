# CNC-Predictive-Maintenance

This repository contains the code and data for a predictive maintenance system for CNC (Computer Numerical Control) machines. The system aims to predict potential failures and maintenance needs to minimize downtime and enhance operational efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Predictive maintenance involves using data analysis tools and techniques to detect anomalies in operations and possible defects in equipment and processes so you can fix them before they result in failure. This project focuses on predictive maintenance for CNC machines, leveraging machine learning algorithms to predict failures based on historical data.

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

To use this project, follow these steps:

1. **Set Up Your Environment**:
    Ensure you have followed the installation instructions to set up your environment.

2. **Run the Streamlit Application**:
    Open a terminal and run the following command to start the Streamlit application:
    ```bash
    streamlit run main.py
    ```

3. **Interact with the Web Interface**:
    Follow the instructions on the web interface to use the application:

    - **Upload Your Dataset**:
        Upload your dataset file in CSV format using the file uploader in the configuration panel on the left side of the interface.

    - **Configure Parameters**:
        Set up the necessary parameters in the configuration panel on the left side of the interface.

    - **Execute the Pipeline**:
        Click the `🚀 Execute All` button to run the entire pipeline, including preprocessing, training, and evaluation.

    - **View the Results**:
        The results, including various plots and evaluations, will be displayed on the right side of the interface.

## Configuration

The configuration settings for running the scripts are defined as follows:

| Configuration             | Description                                                                                                                                                        |
| ------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Model**                 | The machine learning model to be used for prediction.                                                                                                              |
| **Sampler**               | Sampling method to balance the dataset.                                                                                                                            |
| **Sampling Strategy**     | Desired number of samples for each class after resampling. You can specify the exact number of samples for non-anomalous and anomalous classes.            |
| **Future Steps**          | Number of future steps to predict. For example, if set to 3, the model will predict steps $K$, $K+1$, and $K+2$.                                                   |
| **Window Size**           | Size of the sliding window for feature extraction. For example, a window size of 3 means using the past 3 time steps $K-3$, $K-2$, $K-1$ to predict the next step. |
| **Test Size**             | Number of samples to include in the test split. This determines the amount of data reserved for testing the model.                                                 |
| **Seed**                  | Random seed for reproducibility. Setting this ensures that the results are consistent across different runs.                                                       |
| **Correlation Threshold** | Correlation threshold for feature selection. Features with correlation higher than this threshold will be removed to avoid multicollinearity.                      |

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
