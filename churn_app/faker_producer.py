import numpy as np
import pandas as pd
from faker import Faker
from .database import _is_missing_value

fake = Faker()

VALID_DEPARTMENTS = ["Marketing", "Sales", "Engineering", "HR", "Support", "Operations", "Finance"]
VALID_CONTRACTS = ["permanent", "contract"]
DEFAULT_JOBS = {
    "Marketing": ["Marketing Analyst", "Content Creator", "Marketing Manager", "Social Media Specialist"],
    "Sales": ["Account Executive", "Sales Representative", "Sales Engineer", "Sales Manager"],
    "Engineering": ["Software Engineer", "DevOps Engineer", "QA Engineer", "Frontend Developer", "Backend Engineer"],
    "HR": ["HR Coordinator", "HR Business Partner", "Recruiter", "Talent Specialist"],
    "Support": ["Support Specialist", "Support Manager", "Customer Success Lead"],
    "Operations": ["Operations Manager", "Supply Chain Analyst", "Logistics Specialist"],
    "Finance": ["Financial Analyst", "Accountant", "Payroll Specialist", "Controller"],
}


class FakerProducer:
    """Python producer and faker to generate realistic identities and repair unapproved/rejected rows."""

    def __init__(self, seed: int | None = None):
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)
            np.random.seed(seed)

    def generate_name(self) -> tuple[str, str]:
        """Generate a random first and last name."""
        return self.faker.first_name(), self.faker.last_name()

    def repair_rejected_rows(self, rejected_df: pd.DataFrame) -> pd.DataFrame:
        """
        Inspect rejected rows and use Faker to synthesize missing first/last names or resolve duplicate conflicts.
        Returns a DataFrame ready for validation and database insertion.
        """
        if rejected_df.empty:
            return pd.DataFrame()

        repaired_records = []
        for _, row in rejected_df.iterrows():
            record = row.to_dict()
            reason = str(record.get("reason", "")).lower()
            repaired_actions = []

            first_name = record.get("first_name")
            last_name = record.get("last_name")

            # Check if missing first or last name
            missing_first = _is_missing_value(first_name)
            missing_last = _is_missing_value(last_name)

            if "duplicate" in reason or "already exists" in reason:
                # Generate a completely fresh identity to resolve uniqueness collision
                new_first, new_last = self.generate_name()
                record["first_name"] = new_first
                record["last_name"] = new_last
                repaired_actions.append(f"Replaced identity with {new_first} {new_last} (duplicate resolution)")
            else:
                if missing_first:
                    new_first = self.faker.first_name()
                    record["first_name"] = new_first
                    repaired_actions.append(f"Faked first_name: {new_first}")
                if missing_last:
                    new_last = self.faker.last_name()
                    record["last_name"] = new_last
                    repaired_actions.append(f"Faked last_name: {new_last}")

            # Check for range/value issues and repair if trivial
            if "range" in reason or "finite" in reason or "non-numeric" in reason:
                age_val = record.get("age")
                if _is_missing_value(age_val) or not isinstance(age_val, (int, float)) or (isinstance(age_val, (int, float)) and (age_val < 18 or age_val > 99)):
                    record["age"] = int(np.random.randint(22, 60))
                    repaired_actions.append("Fixed age to valid range")

                income_val = record.get("annual_income")
                if _is_missing_value(income_val) or not isinstance(income_val, (int, float)) or (isinstance(income_val, (int, float)) and income_val < 0):
                    record["annual_income"] = float(np.random.randint(45000, 125000))
                    repaired_actions.append("Fixed annual_income to non-negative")

                tenure_val = record.get("tenure_months")
                if _is_missing_value(tenure_val) or not isinstance(tenure_val, (int, float)) or (isinstance(tenure_val, (int, float)) and (tenure_val < 0 or tenure_val > 240)):
                    record["tenure_months"] = int(np.random.randint(1, 120))
                    repaired_actions.append("Fixed tenure_months to valid range")

            # Check for department/contract validity
            dept = record.get("department")
            if _is_missing_value(dept) or dept not in VALID_DEPARTMENTS:
                dept = str(np.random.choice(VALID_DEPARTMENTS))
                record["department"] = dept
                repaired_actions.append(f"Assigned valid department: {dept}")

            contract = record.get("contract_type")
            if _is_missing_value(contract) or contract not in VALID_CONTRACTS:
                contract = str(np.random.choice(VALID_CONTRACTS, p=[0.75, 0.25]))
                record["contract_type"] = contract
                repaired_actions.append(f"Assigned valid contract_type: {contract}")

            job = record.get("job_title")
            if _is_missing_value(job):
                dept_jobs = DEFAULT_JOBS.get(dept, ["Staff Member"])
                record["job_title"] = str(np.random.choice(dept_jobs))
                repaired_actions.append(f"Assigned job_title: {record['job_title']}")

            record["faker_repairs"] = "; ".join(repaired_actions) if repaired_actions else "No repair needed"
            repaired_records.append(record)

        repaired_df = pd.DataFrame(repaired_records)
        # Drop metadata columns before preparing for import
        return repaired_df

    def generate_synthetic_batch(self, count: int = 10, churn_prob: float = 0.25) -> pd.DataFrame:
        """Produce a batch of completely synthetic employee records."""
        records = []
        for _ in range(count):
            dept = str(np.random.choice(VALID_DEPARTMENTS))
            contract = str(np.random.choice(VALID_CONTRACTS, p=[0.7, 0.3]))
            job = str(np.random.choice(DEFAULT_JOBS[dept]))
            age = int(np.random.randint(22, 62))
            tenure = int(np.random.randint(2, 140))
            income = float(np.random.randint(48000, 130000))
            # Synthetic churn signal correlation
            churn_risk = 0.15 + (0.2 if contract == "contract" else 0.0) + (0.15 if tenure < 18 else -0.05) + (0.1 if income < 60000 else -0.05)
            churn_risk = max(0.05, min(0.90, churn_risk))
            churn = 1 if np.random.rand() < churn_risk else 0

            first, last = self.generate_name()
            records.append({
                "first_name": first,
                "last_name": last,
                "age": age,
                "department": dept,
                "contract_type": contract,
                "annual_income": income,
                "job_title": job,
                "tenure_months": tenure,
                "churn": churn,
            })
        return pd.DataFrame(records)
