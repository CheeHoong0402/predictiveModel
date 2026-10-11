# FoodResilience — retail procurement dashboard

Public demonstration of a Malaysian retail forecasting and inventory-planning project.
Visit the [public dashboard](https://foodresilience-procurement.streamlit.app/).
The retailer transactions are **synthetic**, and the dashboard's decision origin is
**17 August 2024**. These outputs are not live procurement instructions or evidence
of real commercial savings.

## Run the dashboard

Use Python 3.12 (3.11 is also tested locally):

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The included, reviewed demonstration artifacts support four stores and six products.
Research source and reports from the completed phases are preserved. Large training
datasets, model binaries, optimisation caches, private uploads, prompts and local
credentials are deliberately excluded. `requirements-research.txt` records the
original research environment; it is not needed to run the dashboard. Earlier
pipeline/integration tests require their local research datasets.

## Model limitations

- Demand and inventory outputs are historical demonstration results. Reported validation
  and savings figures refer to synthetic data and are not commercial performance claims.
- The purchase-price model is quarantined after a leakage audit and failed validation.
  Future price predictions do not influence orders. Displayed price is the current known
  historical purchase cost, not a supplier quote for today.
- Private data uploads are disabled: the previous adapter did not perform real inference
  on uploaded values. It now fails explicitly instead of returning cached demo results.
- The release corrects product IDs and matches displayed demand forecasts to their actual
  forecast origin. Unknown products/dates fail instead of receiving fabricated defaults.
- Restock scheduling is a planning estimate: order on the day after the end-of-day
  origin, arrival after the demonstration's two-day lead-time assumption, then next
  replenishment after the recommended coverage period. Confirm actual lead times,
  incoming orders and inventory before using any recommendation operationally.

## Telegram delivery

1. Revoke any bot token previously pasted into a chat. Generate its replacement in BotFather.
2. In the hosting service's secret settings, set `TELEGRAM_BOT_TOKEN` to the replacement.
   Never commit it, paste it into an issue, or put it in the browser application.
3. Open **Telegram Alerts & Policy**. Choose one item or the selected store's six items.
4. Click **Connect my Telegram**, open the bot link and press **Start**. Return and verify.
5. Review the exact message and click **Send this summary to my Telegram**.

The message includes the historical decision date, item, quantity and measurement unit,
order date, expected arrival, estimated next replenishment date, current purchase price
and estimated purchase cost. No future-price claim is made. No purchase order is placed.

Each browser session must prove ownership through a unique private `/start` challenge.
The app does not accept arbitrary recipient chat IDs. Chat IDs are held only in server
memory for up to one hour and removed on disconnect/restart. Connection links expire
after ten minutes. Delivery is explicit, with a one-minute recipient cooldown. `/stop`
is processed when the dashboard next checks Telegram updates. Disconnect in the
dashboard removes its authorization immediately. There are no scheduled broadcasts.

Run this bot with **one dashboard process / one Telegram update consumer**, without a
separate webhook or polling service. Scaling or unattended daily delivery requires a
durable subscription store and dedicated worker, which this release does not provide.

## Public hosting

The canonical repository is `CheeHoong0402/predictiveModel`. Streamlit requires repository
administrator access, so deployment uses the owner's hosting fork
`Codinggeniusnks/predictiveModel`. Keep that fork synced with the canonical repository.
On Streamlit Community Cloud choose the hosting fork, branch
`main`, entrypoint `app.py`, Python 3.12. Deploy as a public app. Configure the Telegram
secret privately after token rotation, then reboot the app. Public access must be
verified from a signed-out browser; a successful local run is not a deployment.

## Verification

```sh
python -m pip install -r requirements-test.txt
python scripts/security_scan.py
python -m pytest tests/test_public_release.py tests/test_dashboard.py tests/test_recommendation_engine.py tests/test_model_governance.py -q
```

See `reports/security/publication_audit.md` for audit scope and findings, and
`PUBLIC_DATA_MANIFEST.json` for the only data files approved for publication.
