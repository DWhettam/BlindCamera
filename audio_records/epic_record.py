from .audio_record import AudioRecord


class EpicAudioRecord(AudioRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_timestamp(self):
        return self._series['start_timestamp']

    @property
    def stop_timestamp(self):
        return self._series['stop_timestamp']

    @property
    def audio_length(self):
        return self.stop_timestamp - self.start_timestamp
    
    @property
    def narration(self):
        return self._series['narration']

    @property
    def label(self):
        if 'verb_class' in self._series.keys().tolist():
            label = {'verb': self._series['verb_class'], 'noun': self._series['noun_class']}
        else:  # Fake label to deal with the test sets (S1/S2) that dont have any labels
            label = -10000
        return label