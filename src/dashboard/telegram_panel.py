"""Explicit, opt-in Telegram delivery of the visible demonstration decision."""
import os
import streamlit as st
from src.recommendations.telegram_delivery import (
    TelegramClient, TelegramConnections, TelegramError, decision_message,
)


def _token():
    value = os.environ.get('TELEGRAM_BOT_TOKEN')
    if value:
        return value
    try:
        return st.secrets.get('TELEGRAM_BOT_TOKEN')
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return None


@st.cache_resource(show_spinner=False)
def _connections():
    return TelegramConnections(TelegramClient(_token()))


def render_telegram_panel(recommendations, names, units):
    st.subheader('Send your restocking summary')
    st.caption('Historical synthetic demonstration. Sends only when you click Send; no automatic broadcasts.')
    message = decision_message(recommendations, names, units)
    st.code(message, language=None)
    if not _token():
        st.info('Message preview is available. Telegram delivery awaits a rotated bot token in the host’s secret settings.')
        return
    try:
        broker = _connections()
        if st.button('Connect my Telegram'):
            previous = st.session_state.pop('telegram_capability', None)
            if previous:
                broker.disconnect(previous)
            capability, link = broker.begin()
            st.session_state['telegram_capability'] = capability
            st.session_state['telegram_link'] = link
            st.session_state['telegram_verified'] = False
        capability = st.session_state.get('telegram_capability')
        if capability:
            st.link_button('Open FoodResilienceBot and press Start', st.session_state['telegram_link'])
            st.caption('Connect within 10 minutes. Your chat ID stays on the server for at most one hour. Disconnect to remove it sooner.')
            if st.button('I pressed Start — verify connection'):
                st.session_state['telegram_verified'] = broker.verify(capability)
                if not st.session_state['telegram_verified']:
                    st.info('No matching Start message yet. Press Start in Telegram, then verify again.')
            if st.session_state.get('telegram_verified'):
                st.success('Connected to your Telegram account.')
                if st.button('Send this summary to my Telegram', type='primary'):
                    broker.send(capability, message)
                    st.success('Telegram confirmed delivery.')
            if st.button('Disconnect Telegram'):
                broker.disconnect(capability)
                for key in ['telegram_capability', 'telegram_link', 'telegram_verified']:
                    st.session_state.pop(key, None)
                st.rerun()
    except TelegramError as error:
        st.error(str(error))
    st.caption('Use this bot with one dashboard instance only. Delivery is on demand while this app is running.')
