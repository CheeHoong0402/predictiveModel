"""Opt-in, session-bound Telegram delivery. Tokens and chat IDs never go to the browser."""
from datetime import date, timedelta
import math
import re
import secrets
import threading
import time

import requests


class TelegramError(RuntimeError):
    """Safe error text: never include transport exceptions, URLs, tokens or API bodies."""


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Missing {label}") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {label}")
    return result


def restock_plan(rec):
    """End-of-day T: order on T+1; arrival adds lead time; next cycle adds coverage.

    The next replenishment date is a coverage-based planning estimate, not a supplier
    commitment. Preserve fractional quantities (e.g. kg) and the current known price.
    """
    origin = date.fromisoformat(str(rec['decision_date'])[:10])
    decision, inventory = rec['decision'], rec['inventory']
    quantity = _number(decision['order_quantity'], 'quantity')
    price = _number(rec['price']['current_purchase_price'], 'current price')
    if price <= 0:
        raise ValueError('Current price must be positive')
    lead = math.ceil(_number(inventory['lead_time_days'], 'lead time'))
    coverage = math.ceil(_number(decision['coverage_days'], 'coverage'))
    order = origin + timedelta(days=1) if quantity > 0 else None
    arrival = order + timedelta(days=lead) if order else None
    following = arrival + timedelta(days=coverage) if arrival and coverage else None
    return dict(forecast_origin=origin.isoformat(), order_date=order.isoformat() if order else None,
                restock_date=arrival.isoformat() if arrival else None,
                next_replenishment_date=following.isoformat() if following else None,
                quantity=quantity, current_price=price, estimated_purchase_cost=round(quantity*price, 2))


def _label(value):
    return re.sub(r'[\x00-\x1f\x7f]', ' ', str(value))[:130]


def decision_message(recommendations, product_names=None, units=None):
    if not 1 <= len(recommendations) <= 6:
        raise ValueError('Select between one and six items')
    names, units = product_names or {}, units or {}
    lines = ['FoodResilience — HISTORICAL SYNTHETIC DEMO',
             'Planning illustration only; these are not live restock instructions.']
    for rec in recommendations:
        plan = restock_plan(rec)
        sku = rec['product_id']
        lines.extend(['', f"Store: {_label(rec['store_id'])}",
                      f"Decision date (end of day): {plan['forecast_origin']}",
                      f"Item: {_label(names.get(sku, sku))}",
                      f"Action: {_label(rec['decision']['action']).replace('_', ' ')}",
                      f"Restock quantity: {plan['quantity']:g} {_label(units.get(sku, 'units'))}",
                      f"Order date: {plan['order_date'] or 'No order recommended'}",
                      f"Expected restock arrival: {plan['restock_date'] or 'Not scheduled'}",
                      f"Next replenishment (estimate): {plan['next_replenishment_date'] or 'Review inventory before scheduling'}",
                      f"Current known price: RM {plan['current_price']:.2f} / {_label(units.get(sku, 'unit'))}",
                      f"Estimated purchase cost: RM {plan['estimated_purchase_cost']:.2f}"])
    lines += ['', 'Dates assume ordering on T+1, stated lead time, then the recommended coverage period.',
              'Supplier timing is unconfirmed. Future price prediction is quarantined and not used.']
    message = '\n'.join(lines)
    if len(message) > 4000:
        raise ValueError('Message is too long; select fewer items')
    return message


class TelegramClient:
    def __init__(self, token, http=None):
        if not token:
            raise TelegramError('Telegram is not configured by the host.')
        self._token = token
        self._http = http or requests.Session()

    def call(self, method, payload):
        if method not in {'getMe', 'getUpdates', 'sendMessage'}:
            raise TelegramError('Unsupported Telegram operation.')
        try:
            response = self._http.post(
                f'https://api.telegram.org/bot{self._token}/{method}',
                json=payload, timeout=(5, 12), allow_redirects=False)
            if response.status_code != 200:
                raise TelegramError('Telegram request failed. Check the host configuration or try again later.')
            result = response.json()
            if not result.get('ok'):
                raise TelegramError('Telegram could not complete the request.')
            return result['result']
        except TelegramError:
            raise
        except Exception:
            raise TelegramError('Telegram is temporarily unavailable. No delivery has been confirmed.') from None


class TelegramConnections:
    """Single-process broker. Private /start nonce binds a browser session to its sender.

    Keeps only random capabilities, chat IDs and timestamps in memory. Never accepts
    a browser-provided chat ID. Expiry/restart/disconnect removes authorization.
    No polling daemon or unsolicited daily messages. Run only one consumer per bot.
    """
    def __init__(self, client, clock=time.time):
        self.client, self.clock = client, clock
        self._lock = threading.RLock()
        self._pending, self._sessions, self._last_sent = {}, {}, {}
        self._offset = 0
        self._last_poll = 0

    def _prune(self):
        now = self.clock()
        self._pending = {k:v for k,v in self._pending.items() if now-v['created'] < 600}
        self._sessions = {k:v for k,v in self._sessions.items() if now-v['verified'] < 3600}
        self._last_sent = {k:v for k,v in self._last_sent.items() if now-v < 60}

    def begin(self):
        with self._lock:
            self._prune()
            if len(self._pending) + len(self._sessions) >= 1000:
                raise TelegramError('Connection capacity reached. Please try later.')
            me = self.client.call('getMe', {})
            if me.get('username', '').lower() != 'foodresiliencebot':
                raise TelegramError('The configured token does not belong to FoodResilienceBot.')
            capability = secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(24)
            self._pending[capability] = {'nonce': nonce, 'created': self.clock()}
            return capability, f'https://t.me/FoodResilienceBot?start={nonce}'

    def verify(self, capability):
        with self._lock:
            self._prune()
            if capability not in self._pending and capability not in self._sessions:
                raise TelegramError('Connection expired. Start a new connection.')
            self._poll()
            return capability in self._sessions

    def _poll(self):
        """Caller owns the lock. Process pending /start and /stop before each send."""
        now = self.clock()
        if now-self._last_poll < 2:
            return
        self._last_poll = now
        updates = self.client.call('getUpdates', {'offset': self._offset, 'limit': 100,
                                                   'timeout': 0, 'allowed_updates': ['message']})
        for update in updates:
            self._offset = max(self._offset, int(update['update_id']) + 1)
            message = update.get('message', {})
            chat = message.get('chat', {})
            sender = message.get('from', {})
            if (chat.get('type') != 'private' or not chat.get('id')
                    or chat.get('id') != sender.get('id') or sender.get('is_bot', False)):
                continue
            content = message.get('text', '')
            if content == '/stop':
                self._sessions = {k:v for k,v in self._sessions.items() if v['chat'] != chat['id']}
                continue
            for key, pending in list(self._pending.items()):
                if (content == '/start ' + pending['nonce']
                        and pending['created']-1 <= message.get('date', 0) <= now+5):
                    self._sessions[key] = {'chat': chat['id'], 'verified': now}
                    del self._pending[key]
                    break

    def send(self, capability, text):
        if not 1 <= len(text) <= 4000:
            raise TelegramError('Message length is invalid.')
        with self._lock:
            # Process /stop before delivery, when possible; verification also checks expiry.
            self._prune()
            self._poll()
            if capability not in self._sessions:
                raise TelegramError('Connect your own Telegram account first.')
            chat = self._sessions[capability]['chat']
            if self.clock()-self._last_sent.get(chat, -100000) < 60:
                raise TelegramError('Please wait one minute before sending another summary.')
            # Reserve before sending to prevent duplicate sends after ambiguous timeouts.
            self._last_sent[chat] = self.clock()
            result = self.client.call('sendMessage', {'chat_id': chat, 'text': text,
                        'link_preview_options': {'is_disabled': True}, 'protect_content': True})
            if not isinstance(result, dict) or not result.get('message_id'):
                raise TelegramError('Telegram did not confirm delivery.')
            return True

    def disconnect(self, capability):
        with self._lock:
            self._pending.pop(capability, None)
            self._sessions.pop(capability, None)
