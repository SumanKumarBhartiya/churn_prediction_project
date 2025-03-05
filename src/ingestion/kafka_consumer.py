from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'customer_activity',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for msg in consumer:
    print(f"Received: {msg.value}")
