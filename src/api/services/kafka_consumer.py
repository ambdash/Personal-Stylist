from kafka import KafkaConsumer
import json
import logging
from typing import Callable, Dict, Any
import threading
import os

logger = logging.getLogger(__name__)

class KafkaMessageConsumer:
    def __init__(self):
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.running = True

    def start_consumer(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        """Start a consumer for the given topic with a message handler"""
        def consume():
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=f"stylist_bot_{topic}_consumer",
                auto_offset_reset='latest'
            )
            
            self.consumers[topic] = consumer
            
            while self.running:
                try:
                    for message in consumer:
                        if not self.running:
                            break
                        handler(message.value)
                except Exception as e:
                    logger.error(f"Error processing message from {topic}: {e}")

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()

    def stop_all(self):
        """Stop all consumers"""
        self.running = False
        for consumer in self.consumers.values():
            consumer.close() 