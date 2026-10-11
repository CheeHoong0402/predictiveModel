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
    st.caption('Historical synthetic demonstration. Manual send is available below; a daily 8:00 AM Malaysia-time digest can be enabled for one private chat through repository secrets.')
    message = decision_message(recommendations, names, units)
    st.code(message, language=None)
    if not _token():
        st.info('Message preview is available. Telegram delivery awaits a rotated bot token in the host’s secret settings. Daily delivery also needs private repository secrets.')
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
            st.caption('Connect within 10 minutes. The dashboard keeps your chat ID in memory for at most one hour; daily delivery separately requires you to save it as a repository secret.')
            if st.button('I pressed Start — verify connection'):
                st.session_state['telegram_verified'] = broker.verify(capability)
                if not st.session_state['telegram_verified']:
                    st.info('No matching Start message yet. Press Start in Telegram, then verify again.')
            if st.session_state.get('telegram_verified'):
                st.success('Connected to your Telegram account.')
                st.caption('For daily delivery to this private chat, save this verified chat ID as the TELEGRAM_CHAT_ID repository secret:')
                st.code(str(broker.private_chat_id(capability)), language=None)
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
    st.caption('Manual delivery uses this session. The separate daily schedule uses the private repository secrets.')
