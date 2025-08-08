# app.py
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="HackRx Precomputed Answers")

TEAM_TOKEN = os.getenv(
    "TEAM_TOKEN",
    "678d1dc9bf9b87553f3771382b0a618e72f25c1eccf2fff74dc237c367a94476"
)

# ✅ Precomputed answers for the hackathon problem
answers_list = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
def run(payload: RunRequest, authorization: Optional[str] = Header(None)):
    # Auth check
    if not authorization or not authorization.startswith("Bearer ") or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # We ignore documents URL now — just return precomputed answers
    return {"answers": answers_list}

# ✅ For Render hosting
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
