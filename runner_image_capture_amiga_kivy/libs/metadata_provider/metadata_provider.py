import asyncio

from PIL import Image
from PIL.ExifTags import Base
import json
from amiga_client.amiga_client import AmigaClient


class MetadataProvider:
    def __init__(self, amiga_client, logger=None):
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