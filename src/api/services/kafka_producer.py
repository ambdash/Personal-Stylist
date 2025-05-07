from kafka import KafkaProducer
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

class KafkaMessageProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    async def send_message(self, topic: str, message: Dict[str, Any]):
        """Send message to Kafka topic"""
        try:
            future = self.producer.send(topic, message)
            self.producer.flush()
            record_metadata = future.get(timeout=10)
            logger.info(f"Message sent to topic {topic} at {record_metadata.timestamp}")
        except Exception as e:
            logger.error(f"Error sending message to Kafka: {e}")
            raise 