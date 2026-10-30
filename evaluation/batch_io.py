"""Small, bounded retry helpers for real A3 batch state and console output."""
import errno
import json
import os
import sys
import time


def save_state(path, report, attempts=5):
    """Use native container storage for the authoritative state when available."""
    payload=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    temporary=path.with_suffix(path.suffix+'.tmp')
    for attempt in range(attempts):
        try:
            with temporary.open('w',encoding='utf-8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            return
        except OSError as error:
            if error.errno not in (errno.ENOMEM,errno.EIO,errno.EAGAIN) or attempt+1==attempts:
                raise
            print(f'A3 state write retry {attempt+1}/{attempts}: errno={error.errno}',file=sys.__stderr__,flush=True)
            time.sleep(min(2**attempt,8))


class Tee:
    """Keep a real file log while retaining Docker console visibility."""
    def __init__(self,primary,console):
        self.primary,self.console=primary,console

    def write(self,text):
        self.primary.write(text)
        self.primary.flush()
        try:
            self.console.write(text)
            self.console.flush()
        except (OSError,ValueError):
            pass
        return len(text)

    def flush(self):
        self.primary.flush()
        try:
            self.console.flush()
        except (OSError,ValueError):
            pass

    def isatty(self):
        return False
