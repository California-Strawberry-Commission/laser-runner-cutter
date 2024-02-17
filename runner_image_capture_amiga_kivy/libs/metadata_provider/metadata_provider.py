import asyncio
import os
from google.protobuf import json_format
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_client import EventClient
from PIL import Image
from PIL.ExifTags import GPS
    
class MetadataProvider:
    _last_service_messages = {}
    _subscription_tasks = []
    _config = None
    
    def __init__(self, config):
        self._load_config(config)
    
    def _load_config(self, config):
        """Loads a standard Farm-NG service configuration, passed as a JSON dict"""
        self._config = json_format.ParseDict(config, EventServiceConfigList())

    async def init_clients(self):
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
        self._last_service_messages[service_name] = {}
        async for event, msg in event_client.subscribe(subscription):
            print(msg)
            self._last_service_messages[service_name][event.uri.path] = msg

    def get_message(self, service, path):
        """Returns the latest message from the given path and service"""
        if (s := self._last_service_messages.get(service)) is not None:
            if (p := s.get(path)) is not None:
                return p
        return None
        
    def add_exif(self, file):
        """Opens the passed file, sets exif metadata, and saves it https://exiv2.org/tags.html
        
        Note: EXIF seems to be a bit of a mess and python support isn't great, so fields are
        just stored as a JSON dict in the UserComment
        """
        im = Image.open(file)
        exif = im.getexif()
        
        dataDict = {}
        print(self._last_service_messages)
        
        if m := self.get_message("gps", "/pvt"):
            print("Set GPS location information")
            dataDict["gps"] = m
        
        # Put JSON in Usercomment
        exif[0x9286] = str(dataDict) 
        
        im.save(file + "exif.png", exif=exif)
    
# Testing - do not run this file as main.
if __name__ == "__main__":
    m = MetadataProvider({
        "configs": [
            {
                "name": "gps",
                "port": 3001,
                "host": "129.65.121.182",
                "log_level": "INFO",
                "subscriptions": [
                    {
                        "uri": {
                            "path": "*",
                            "query": "service_name=gps"
                        },
                        "every_n": 1
                    }
                ]
            }
        ]
    })
        
    async def main():
        async def exif_trigger():
            await asyncio.sleep(5)
            print("Triggering exif mod")
            m.add_exif("/mnt/c/Users/t-dchmiel/Projects/laser-runner-cutter/runner_image_capture_amiga_kivy/libs/metadata_provider/strawberry.png")

        asyncio.create_task(exif_trigger())
        await m.init_clients()
        await asyncio.gather(*m._subscription_tasks)
    
    asyncio.new_event_loop().run_until_complete(main()) 