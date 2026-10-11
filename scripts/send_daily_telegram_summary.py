"""Send the reviewed historical demonstration restocking digest to one private chat.

Run by the GitHub Actions schedule, independently of Streamlit's sleeping process.
No credentials or recipient identifiers are accepted on the command line or logged.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.config import PRODUCT_LABELS, PRODUCT_UNITS, STORE_LABELS
from src.recommendations.engine import generate_final_recommendation, load_cached_phase_artifacts
from src.recommendations.telegram_delivery import TelegramClient, TelegramError, decision_message


def build_daily_messages(root=ROOT, today=None):
    """One Telegram-sized message per store; never represent 2024 data as live."""
    today = today or datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).date()
    artifacts = load_cached_phase_artifacts(root=Path(root))
    messages = []
    for store_id in STORE_LABELS:
        recommendations = [generate_final_recommendation(
            store_id, sku, root=Path(root), cached_artifacts=artifacts,
        ) for sku in PRODUCT_LABELS]
        eligible = [rec for rec in recommendations if float(rec['decision']['order_quantity']) > 0]
        header = f'Daily demo digest — {today.isoformat()} (Malaysia time)\n{STORE_LABELS[store_id]}\n'
        if eligible:
            body = decision_message(eligible, PRODUCT_LABELS, PRODUCT_UNITS)
        else:
            body = ('FoodResilience — HISTORICAL SYNTHETIC DEMO\n'
                    'No restock quantity in this saved demonstration. No live order is implied.')
        text = header + body
        if len(text) > 4000:
            raise ValueError('Daily store summary exceeds Telegram message limit')
        messages.append(text)
    return messages


def send_daily_messages(messages, token, chat_id, client_factory=TelegramClient):
    if not token:
        raise TelegramError('Missing TELEGRAM_BOT_TOKEN repository secret.')
    if not chat_id or not str(chat_id).lstrip('-').isdigit():
        raise TelegramError('Missing or invalid TELEGRAM_CHAT_ID repository secret.')
    client = client_factory(token)
    identity = client.call('getMe', {})
    if identity.get('username', '').lower() != 'foodresiliencebot':
        raise TelegramError('Configured token does not belong to FoodResilienceBot.')
    for message in messages:
        result = client.call('sendMessage', {
            'chat_id': int(chat_id), 'text': message,
            'link_preview_options': {'is_disabled': True}, 'protect_content': True,
        })
        if not isinstance(result, dict) or not result.get('message_id'):
            raise TelegramError('Telegram did not confirm delivery.')
    return len(messages)


def main():
    parser = argparse.ArgumentParser(description='Daily historical demo Telegram summary')
    parser.add_argument('--dry-run', action='store_true', help='Validate summaries without sending')
    args = parser.parse_args()
    try:
        messages = build_daily_messages()
        if args.dry_run:
            print(f'Validated {len(messages)} historical demonstration store summaries; no message sent.')
            return 0
        count = send_daily_messages(messages, os.environ.get('TELEGRAM_BOT_TOKEN'),
                                    os.environ.get('TELEGRAM_CHAT_ID'))
        print(f'Telegram confirmed {count} daily store summaries.')
        return 0
    except (TelegramError, ValueError) as error:
        print(f'Daily summary failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
