# Baatwise

Baatwise is exploring a multilingual conversation follow-through copilot for small education and admissions counseling teams in India. The product thesis is to prepare a counselor-reviewable case brief from confirmed, de-identified learner context and draft a follow-up in the learner’s preferred language.

This repository contains an early evaluation build of that workflow. It is not a production service, and the initial customer segment remains a hypothesis. The build has no accounts, shared team workspace, persistent learner records, WhatsApp or CRM integrations, or send-message capability.

## What the app does

- Collects a learner goal, current stage, preferred reply language, confirmed constraints, next step, and de-identified conversation notes.
- Sends those case fields to Anthropic’s Claude API after the counselor confirms permission to use the notes.
- Produces a case brief with confirmed context, items to confirm, a suggested next step, and an editable follow-up draft.
- Keeps the case reference local to the Streamlit session and does not include it in the Claude request.
- Lets the counselor edit the result and download it as a text file. The app never sends a message to a learner.
- Offers Claude Sonnet 5.5 by default and Claude Haiku 4.5 as the lower-cost option.

Claude can make mistakes. The prompt tells it to use only supplied facts, separate missing details, and avoid making eligibility or admissions decisions. A counselor must verify every fact and decide what to use.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-anthropic-api-key"
streamlit run app.py
```

Alternatively, create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

The secrets file should not be committed. Anthropic API usage is billed separately from Claude subscriptions or startup credits.

## Data handling and limitations

The app stores form values and generated text in Streamlit session state only; it has no Baatwise database. The case reference is shown in the interface but is not sent to Anthropic. When the counselor generates a draft, the other case fields are sent to Anthropic’s API for processing. Anthropic handles those requests under its own policies and account settings.

Use de-identified, permissioned sample data for evaluation. Do not enter names, contact details, identity documents, financial or health information, or other sensitive data. This evaluation build does not provide authentication, durable storage, team access controls, retention controls, audit logs, or production safeguards. Review applicable service terms and privacy requirements before any real-world pilot.

## Product validation

The first segment to investigate is small education and admissions counseling teams in India. Customer demand, workflow fit, integrations, pricing, and time saved have not been validated. Next steps are counselor interviews, testing one repeat follow-up workflow, and evaluating correctness and ease of correction before making outcome claims.
