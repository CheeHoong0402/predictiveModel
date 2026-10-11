# Public release privacy and security review

Reviewed on 11 October 2026, before the initial GitHub publication.

## Scope and results

- The original project's 579 text files were scanned for common Telegram, GitHub,
  OpenAI and cloud credentials, private keys, embedded URL credentials, email
  addresses, secret assignments and personal filesystem paths. No matching live
  credentials, private keys or email addresses were found in those files.
- Personal Windows paths were found in 17 files (prompts, reports and three research
  modules). Prompts were excluded and public copies were sanitised. The original
  research folder was not altered.
- The bot token pasted into the conversation is exposed. It was never copied into
  this release, used for API calls, or placed in logs. Revoke it in BotFather and
  configure a replacement only in hosting secrets before enabling delivery.
- The complete staged release is scanned again by `scripts/security_scan.py`,
  including decoded string columns in every published Parquet file. Only the seven
  synthetic demonstration files in `PUBLIC_DATA_MANIFEST.json` are allowed. Customer,
  contact, address, credential and chat-ID columns are prohibited.
- Raw/interim/processed training data, model binaries, training caches, private
  uploads, environment files, test logs, local prompts and scratch scripts are not
  published. Reports concern synthetic operations and public-source research.
- The repository previously had no commits. Publication uses a new, reviewed history;
  there is no old Git history to scrub.

## Release fixes

- Corrected dashboard product IDs to the actual six-product catalogue.
- Matched demand and inventory records to the displayed forecast origin. Unknown
  products/dates and missing forecasts now fail instead of returning invented defaults.
- Regenerated the release's final recommendations and executive summaries after that
  correction. Original artifacts remain unchanged.
- Disabled private uploads and the custom inference adapter: the old adapter returned
  cached demonstration recommendations regardless of uploaded quantities.
- Kept CORS and XSRF protections enabled and suppressed public exception details.
- Preserved price-model quarantine. Messages use current known historical prices.
- Added private `/start` challenge verification, session-bound Telegram recipients,
  expiring in-memory connection state, explicit send controls and recipient cooldown.
  Transport exceptions are redacted; the verified chat ID is shown only in its own
  browser session for private scheduled-job configuration. The token is never shown.

## Validation

- Original Phase 8/9, dashboard, governance and recommendation prechecks: **157 passed**.
- Original Phase 5–10 regression suite after preparation: **177 passed**.
- Public release dashboard, governance, recommendation and Telegram suite: **96 passed**.
- Original project preservation: **865 existing files unchanged by SHA-256 comparison**.
- Public release secret/PII-pattern scan: **no findings** before initial publication.
- Telegram tests use fake transport; live delivery requires token rotation and a
  verified recipient. Test success does not claim a real Telegram message was sent.

## Limits and operating contract

Pattern scanning and schema review cannot prove the absence of every possible secret
or identifier. Future files and data require review before publication. The public
dashboard is a historical synthetic demonstration, not a live commercial deployment.
The price model remains unvalidated. Restock dates are explicit coverage/lead-time
estimates, not confirmed supplier appointments. Private upload inference is unavailable.
The dashboard runs one update consumer and can send on demand. A separate GitHub
Actions job can send the saved historical demo summary at approximately 8:00 AM
Malaysia time to one private chat after rotated-token and verified-chat-ID secrets
are configured on the hosting fork. It does not consume Telegram updates. Restarting
the app clears dashboard connections. Hosting retains its own service logs; the
application does not persist Telegram identities or user-uploaded retailer data.
