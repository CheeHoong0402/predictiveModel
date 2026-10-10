from copy import deepcopy
from pathlib import Path
import pytest
import pandas as pd

from src.recommendations.telegram_delivery import (
    TelegramClient, TelegramConnections, TelegramError, decision_message, restock_plan,
)


@pytest.fixture
def rec():
    return {'decision_date': '2024-08-17', 'store_id': 'S1', 'product_id': 'P1',
            'decision': {'order_quantity': 20.5, 'coverage_days': 7, 'action': 'BUY_NOW'},
            'inventory': {'lead_time_days': 2},
            'price': {'current_purchase_price': 7.12, 'predicted_future_price': 99999}}


def test_message_fields_and_end_of_day_dates(rec):
    plan = restock_plan(rec)
    assert plan['order_date'] == '2024-08-18'
    assert plan['restock_date'] == '2024-08-20'
    assert plan['next_replenishment_date'] == '2024-08-27'
    text = decision_message([rec], {'P1':'Chicken'}, {'P1':'kg'})
    for value in ['Chicken', '20.5 kg', 'RM 7.12', '2024-08-27', 'SYNTHETIC DEMO']:
        assert value in text
    assert '99999' not in text


def test_no_order_has_no_delivery_date(rec):
    rec['decision']['order_quantity'] = 0
    assert restock_plan(rec)['restock_date'] is None
    assert restock_plan(rec)['next_replenishment_date'] is None


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), None])
def test_invalid_quantities_rejected(rec, bad):
    rec['decision']['order_quantity'] = bad
    with pytest.raises(ValueError): restock_plan(rec)


class FakeClient:
    def __init__(self):
        self.updates, self.sent = [], []
    def call(self, method, payload):
        if method == 'getMe': return {'username':'FoodResilienceBot'}
        if method == 'getUpdates': return [u for u in self.updates if u['update_id'] >= payload['offset']]
        if method == 'sendMessage':
            self.sent.append(payload)
            return {'message_id':1}


def update(nonce, clock, chat=123, kind='private', sender=123, update_id=1):
    return {'update_id':update_id, 'message':{'text':'/start '+nonce, 'date':clock,
            'chat':{'id':chat,'type':kind}, 'from':{'id':sender}}}


def test_private_nonce_binding_isolates_users_and_deduplicates():
    now=[1000]; client=FakeClient(); broker=TelegramConnections(client, lambda:now[0])
    first, link=broker.begin(); second, other=broker.begin()
    nonce=link.split('=')[-1]
    client.updates=[update(nonce,1000)]
    assert broker.verify(first)
    assert not broker.verify(second)
    with pytest.raises(TelegramError): broker.send(second,'Blocked')
    broker.send(first,'Allowed')
    assert client.sent[0]['chat_id']==123
    assert 'parse_mode' not in client.sent[0]
    with pytest.raises(TelegramError): broker.send(first,'Duplicate')
    broker.disconnect(first)
    with pytest.raises(TelegramError): broker.send(first,'Disconnected')


@pytest.mark.parametrize('kind,sender,age', [('group',123,0),('private',456,0),('private',123,-10)])
def test_group_spoofed_or_old_start_cannot_connect(kind,sender,age):
    client=FakeClient();broker=TelegramConnections(client,lambda:1000)
    cap,link=broker.begin()
    client.updates=[update(link.split('=')[-1],1000+age,kind=kind,sender=sender)]
    assert not broker.verify(cap)


def test_expiry_and_stop_remove_access():
    now=[1000];client=FakeClient();broker=TelegramConnections(client,lambda:now[0])
    cap,link=broker.begin();now[0]=1601
    with pytest.raises(TelegramError):broker.verify(cap)
    cap,link=broker.begin();client.updates=[update(link.split('=')[-1],1601)]
    assert broker.verify(cap)
    now[0]+=61
    stop=update('',now[0],update_id=2);stop['message']['text']='/stop'
    client.updates.append(stop)
    with pytest.raises(TelegramError):broker.send(cap,'Blocked after stop')
    assert not client.sent


def test_network_exception_never_exposes_credentials():
    class Broken:
        def post(self,*args,**kwargs): raise RuntimeError('SENSITIVE transport URL')
    with pytest.raises(TelegramError) as result:
        TelegramClient('test-only',Broken()).call('getMe',{})
    assert 'SENSITIVE' not in str(result.value)
    assert 'test-only' not in str(result.value)


def test_six_item_message_fits_telegram_limit():
    from src.dashboard.config import PRODUCT_LABELS, PRODUCT_UNITS
    from src.recommendations.engine import generate_final_recommendation
    recs=[generate_final_recommendation('STR_JHR_003',sku) for sku in PRODUCT_LABELS]
    text=decision_message(recs,PRODUCT_LABELS,PRODUCT_UNITS)
    assert len(text)<4000


def test_dashboard_forecasts_match_selected_origin():
    from src.recommendations.engine import generate_final_recommendation, load_cached_phase_artifacts
    cache=load_cached_phase_artifacts()
    rec=generate_final_recommendation('STR_JHR_003','PRD_CHK_01',cached_artifacts=cache)
    d=cache['calibrated_demand']
    row=d[(d.store_id=='STR_JHR_003') & (d.product_id=='PRD_CHK_01') & (d.horizon==7)
          & (pd.to_datetime(d.forecast_origin)==pd.Timestamp(rec['decision_date']))].iloc[0]
    assert rec['demand']['7d']['p50']==round(float(row.calibrated_p50),1)
    changed=deepcopy(cache)
    changed['calibrated_demand'].loc[pd.to_datetime(d.forecast_origin)!=pd.Timestamp(rec['decision_date']),'calibrated_p50']=999999
    after=generate_final_recommendation('STR_JHR_003','PRD_CHK_01',cached_artifacts=changed)
    assert after['demand']==rec['demand']


def test_unknown_product_cannot_receive_fabricated_recommendation():
    from src.recommendations.engine import generate_final_recommendation
    with pytest.raises(ValueError,match='No audited inventory'):
        generate_final_recommendation('STR_JHR_003','UNKNOWN')


def test_app_starts_and_has_no_upload_widget():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=60)
    assert not app.exception, [x.message for x in app.exception]
    assert len(app.get('file_uploader'))==0
    assert any('rotated bot token' in x.value for x in app.info)
    import json
    summary=json.loads((Path(__file__).parents[1]/'reports/recommendations/executive_summary.json').read_text())
    urgent=next(x for x in app.metric if 'Urgent Reorders' in x.label)
    assert urgent.value==f"{summary['urgent_actions_count']} items"
