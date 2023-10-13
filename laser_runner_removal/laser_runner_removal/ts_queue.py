import heapq

class TsQueue:
    def __init__(self, max_length):
        self.datums = []
        self.max_length = max_length

    def empty(self): 
        self.datums = []

    def add(self, timestamp, data):
        heapq.heappush(self.datums, (timestamp, data))
        if len(self.datums) > self.max_length:
            heapq.heappop(self.datums)

    def get_ts(self, ts):
        closest_datum = None
        closest_ts = None
        closest_diff = float('inf')

        for timestamp, datum in self.datums:
            time_diff = abs(ts - timestamp)
            if time_diff < closest_diff:
                closest_diff = time_diff
                closest_datum = datum
                closest_ts = timestamp
        
        return closest_ts, closest_datum