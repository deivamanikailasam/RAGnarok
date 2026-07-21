# SSO Enablement Runbook

This runbook explains how to enable Single Sign-On (SSO) for a customer.

## Availability by Plan

SSO with SAML is available on the Growth and Enterprise plans. It is not available on the
Consumer or Pro plans. SCIM user provisioning is Enterprise-only.

## Steps to Enable SAML SSO

1. Confirm the customer is on the Growth or Enterprise plan.
2. In the admin console, open Security, then Single Sign-On.
3. Upload the identity provider (IdP) metadata XML, or enter the SSO URL and certificate.
4. Map the email attribute to the IdP NameID.
5. Send the customer a test link and verify a successful login before enforcing SSO.

## Troubleshooting

If login fails with "attribute mismatch", re-check the email attribute mapping in step 4. If the
certificate is rejected, confirm it has not expired and is the signing (not encryption) cert.
