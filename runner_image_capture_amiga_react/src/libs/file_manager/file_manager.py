import os
import cv2
from natsort import natsorted


class FileManager:
    def save_frame(self, frame, directory, prefix):
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(
            directory, self._get_next_filename_with_prefix(directory, prefix)
        )
        cv2.imwrite(file_path, frame)
        return file_path

    def _get_next_filename_with_prefix(self, directory, prefix):
        last_file = self._find_last_filename_with_prefix(directory, prefix)
        next_id = (
            0
            if last_file is None
            else self._get_integer_in_filename(last_file, prefix) + 1
        )
        return f"{prefix}{next_id}.png"

    def _find_last_filename_with_prefix(self, directory, prefix):
        # TODO: cache files
        files = [
            f
            for f in os.listdir(directory)
            if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f))
        ]

        if not files:
            return None

        sorted_files = natsorted(files)
        last_file = sorted_files[-1]
        return last_file

    def _get_integer_in_filename(self, filename, prefix):
        # Remove prefix
        filename = filename[len(prefix) :] if filename.startswith(prefix) else filename
        # Remove extension
        root, _ = os.path.splitext(filename)
        try:
            return int(root)
        except ValueError:
            return -1
