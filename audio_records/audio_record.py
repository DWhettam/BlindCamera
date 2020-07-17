class AudioRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def segment_name(self):
        return NotImplementedError()

    @property
    def untrimmed_video_name(self):
        return NotImplementedError()

    @property
    def start_timestamp(self):
        return NotImplementedError()

    @property
    def stop_timestamp(self):
        return NotImplementedError()

    @property
    def audio_length(self):
        return NotImplementedError()
    
    @property
    def narration(self):
        return self._series['narration']

    @property
    def label(self):
        return NotImplementedError()