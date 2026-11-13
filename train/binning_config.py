import numpy as np
BINNING_CONFIG = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "bins": [-np.inf,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            1.0, np.inf
        ],
        "labels": [
            "<10%",
            "10-30%",
            "30-50%",
            "50-70%",
            "70-90%",
            "90-100%",
            ">100%"
        ]
    },
    "age": {
        "bins": [-np.inf,
            30,
            40,
            50,
            60,
            70, np.inf
        ],
        "labels": [
            "<30",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70+"
        ]
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "bins": [-np.inf,
            1,
            2, np.inf
        ],
        "labels": [
            "0",
            "1",
            "2+"
        ]
    },
    "DebtRatio": {
        "bins": [-np.inf,
            0.2,
            0.4,
            0.6,
            1.0,
            5.0, np.inf
        ],
        "labels": [
            "<20%",
            "20-40%",
            "40-60%",
            "60-100%",
            "1-5x",
            ">5x"
        ]
    },
    "MonthlyIncome": {
        "bins": [-np.inf,
            2500,
            5000,
            7500,
            10000,
            15000, np.inf
        ],
        "labels": [
            "<2.5k",
            "2.5k-5k",
            "5k-7.5k",
            "7.5k-10k",
            "10k-15k",
            ">15k"
        ]
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "bins": [-np.inf,
            3,
            6,
            10,
            15, np.inf
        ],
        "labels": [
            "0-2",
            "3-5",
            "6-9",
            "10-14",
            "15+"
        ]
    },
    "NumberOfTimes90DaysLate": {
        "bins": [-np.inf,
            1,
            2, np.inf
        ],
        "labels": [
            "0",
            "1",
            "2+"
        ]
    },
    "NumberRealEstateLoansOrLines": {
        "bins": [-np.inf,
            1,
            2,
            3, np.inf
        ],
        "labels": [
            "0",
            "1",
            "2",
            "3+"
        ]
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "bins": [-np.inf,
            1,
            2, np.inf
        ],
        "labels": [
            "0",
            "1",
            "2+"
        ]
    },
    "NumberOfDependents": {
        "bins": [-np.inf,
            1,
            2,
            3, np.inf
        ],
        "labels": [
            "0",
            "1",
            "2",
            "3+"
        ]
    }
}