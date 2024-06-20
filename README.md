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


## Usage

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
