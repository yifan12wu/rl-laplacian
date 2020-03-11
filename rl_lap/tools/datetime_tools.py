import datetime
import re

def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now
