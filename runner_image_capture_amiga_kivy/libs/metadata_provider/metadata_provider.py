import asyncio

from PIL import Image
from PIL.ExifTags import Base
import json
from amiga_client.amiga_client import AmigaClient


class MetadataProvider:
    def __init__(self, amiga_client, logger=None, ):
        self.amiga_client = amiga_client
        self.logger = logger
    
    def _log(self, str):
        if self.logger:
            self.logger.info(str)
            
    def get_exif(self, file):
        """Loads JSON-encoded metadata from MakerNote"""
        im = Image.open(file)
        exif = im.getexif()
        
        if not Base.MakerNote in exif:
            return None
        
        return json.loads(exif[Base.MakerNote])

    def add_exif(self, file, overwrite = True):
        """Opens the passed file, sets exif metadata, and saves it https://exiv2.org/tags.html"""
        self._log(f"Add EXIF to {file}")

        im = Image.open(file)
        exif = im.getexif()
        
        exif[Base.MakerNote] = json.dumps(self.amiga_client.get_cache())
        
        file_path = file if overwrite else file + "exif.png"
        im.save(file_path, exif=exif)
        
        return file_path
        
        
# Testing - do not run this file as main.
if __name__ == "__main__":
    c = AmigaClient({
        "configs": [
            {
                "name": "gps",
                "port": 3001,
                "host": "129.65.121.182",
                "log_level": "INFO",
                "subscriptions": [
                    {
                        "uri": {
                            "path": "/pvt",
                            "query": "service_name=gps"
                        },
                        "every_n": 1
                    },
                ]
            },
            {
                "name": "filter",
                "port": 20001,
                "host": "129.65.121.182",
                "log_level": "INFO",
                "subscriptions": [
                    {
                        "uri": {
                            "path": "/state",
                            "query": "service_name=filter"
                        },
                        "every_n": 1
                    }
                ]
            }
        ]
    })
    
    m = MetadataProvider(c)
        
    async def main():
        async def exif_trigger():
            await asyncio.sleep(2)
            print("Triggering exif mod")
            m.add_exif("/mnt/c/Users/t-dchmiel/Projects/laser-runner-cutter/runner_image_capture_amiga_kivy/libs/metadata_provider/strawberry.png")
            await asyncio.sleep(2)
            print(m.get_exif("/mnt/c/Users/t-dchmiel/Projects/laser-runner-cutter/runner_image_capture_amiga_kivy/libs/metadata_provider/strawberry.png")["gps"]["/pvt"]["latitude"])

        asyncio.create_task(exif_trigger())
        c.init_clients()
        await asyncio.gather(*c._subscription_tasks)
    
    asyncio.new_event_loop().run_until_complete(main()) 