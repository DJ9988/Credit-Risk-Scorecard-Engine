from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

class BorrowerProfile(BaseModel):
    model_config = ConfigDict(populate_by_name = True)
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0.0, description="Total balance on credit cards / limit")
    age: int = Field(..., ge=18, le=120, description="Borrower age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(0, ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0.0, description="Debt payments / monthly gross income")
    MonthlyIncome: Optional[float] = Field(default=None, ge=0.0, description="Monthly gross income")
    NumberOfOpenCreditLinesAndLoans: int = Field(0, ge=0, description="Number of active loans and credit cards")
    NumberOfTimes90DaysLate: int = Field(0, ge=0, description="Number of times 90+ days late")
    NumberRealEstateLoansOrLines: int = Field(0, ge=0, description="Mortgages and real estate lines")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(0, ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: Optional[int] = Field(default=None, ge=0, description="Number of family dependents")


class ScoreResponse(BaseModel):
    credit_score: int = Field(..., description = "Credit Score of the individual")
    decision: str = Field(..., description = "Approved / Rejected")
    default_probability: float = Field(..., description = "Probability of Default")
    adverse_actions_reasons: List = Field(..., description = "Reasons for Adverse Actions")