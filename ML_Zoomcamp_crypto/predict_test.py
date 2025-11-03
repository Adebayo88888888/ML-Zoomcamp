import requests

wallet_id = "0x94a5458bad9b21190f42a392512845707d91182f"


host = 'trader-service-env.eba-i2z6eidw.eu-west-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

trader = {
    'active_weeks': 92,
    'total_volume': 761,
    'trader_activity_status' : 'Middle Value Trader',
    'trader_weekly_frequency_status': 'OG',
    'tx_count_365d': 20,
    'community_member': 'not_sure'
}

response = requests.post(url, json=trader).json()
print(response)


prob = response['good_trader_probability']

if prob > 0.9:
    print(f"{wallet_id} is identified as a top-tier trader (High-Value Segment).")
    print("> Triggering VIP reward flow: lower trading fees, early access to beta features, or exclusive staking rewards.\n")

elif prob > 0.6:
    print(f"{wallet_id} classified as an active and promising trader.")
    print("> Sending personalized trading incentive: e.g., gas fee rebate or bonus for next 5 trades.\n")

else:
    print(f"{wallet_id} appears inactive or low-engagement.")
    print("> Initiating reactivation campaign: educational content, referral bonuses, or 'we miss you' notification.\n")
