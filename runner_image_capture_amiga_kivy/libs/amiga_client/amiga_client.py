import asyncio
from google.protobuf import json_format
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_client import EventClient

class AmigaClient:
    _service_message_cache = {}
    _subscription_tasks = []
    _config = None
    
    def __init__(self, config):
        self._load_config(config)
    
    def _load_config(self, config):
        """Loads a standard Farm-NG service configuration, passed as a JSON dict"""
        self._config = json_format.ParseDict(config, EventServiceConfigList())

    def init_clients(self):
        """Initializes eventclients and tasks"""
        # Seperate task initialization from constructor to capture asyncio event loop
        for config in self._config.configs:
            # create the event client
            client = EventClient(config=config)
            for subscription in config.subscriptions:
                task = asyncio.create_task(self._task_store_subscription_messages(config.name, subscription, client))
                self._subscription_tasks.append(task)

    async def _task_store_subscription_messages(self, service_name, subscription, event_client):
        """Long-running task which stores most recent channel messages"""
        self._service_message_cache[service_name] = {}
        async for event, msg in event_client.subscribe(subscription):
            print(event.uri.path)
            msg_dict = json_format.MessageToDict(msg)  # Store keep local record as plain JSON
            self._service_message_cache[service_name][event.uri.path] = msg_dict

    def get_message(self, service, path):
        """Returns the last cached message from the given path and service"""
        if (s := self._service_message_cache.get(service)) is not None:
            if (p := s.get(path)) is not None:
                return p
        return None

    def get_cache(self):
        return self._service_message_cache