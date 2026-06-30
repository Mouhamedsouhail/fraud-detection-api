# Security Policy

SentinelPay is a demo/reference fraud-detection service. It is not a PCI-DSS compliant production system as-is.

## Reporting

Open a private security advisory in GitHub or contact the repository owner directly if you find a vulnerability.

## Operational Notes

- Never commit `.env`, API keys, model artifacts, or raw transaction datasets.
- Put the API behind authentication before exposing it beyond local development.
- Treat risk scores as decision support, not final fraud determinations.
- Add audit logging, access controls, encryption, and retention policies before handling production payment data.
