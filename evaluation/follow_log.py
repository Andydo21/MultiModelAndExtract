"""Retry transient Windows bind-mount reads without interrupting the A3 worker."""
import argparse
import os
import sys
import time
from pathlib import Path


def follow(path, output, error_output, *, last_lines=20, once=False):
    offset=None
    last_error=0.0
    while True:
        try:
            # Reopen each poll: a failed shared-filesystem handle must not be reused.
            with path.open('rb') as source:
                size=os.fstat(source.fileno()).st_size
                initial=offset is None
                if initial:
                    source.seek(max(0,size-65536))
                    data=source.read()
                    data=b''.join(data.splitlines(keepends=True)[-last_lines:])
                else:
                    source.seek(0 if size<offset else offset)
                    data=source.read(65536)
                next_offset=source.tell()
            if data:
                output.write(data)
                output.flush()
            offset=next_offset
            if once:
                return
            if data and not initial:
                continue
        except OSError as error:
            if once:
                raise
            now=time.monotonic()
            if now-last_error>=30:
                detail='memory allocation failed while reading the shared log' if error.errno==12 else 'shared log access failed'
                error_output.write((f'Log viewer retrying: {detail} ({type(error).__name__}, errno={error.errno}). '
                    'This viewer does not verify A3 extraction health.\n').encode())
                error_output.flush()
                last_error=now
        time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--container-stdout',action='store_true')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.container_stdout:
        import fcntl
        # Container-local lock, independent of the Windows-mounted log file.
        with open('/tmp/a3-log-viewer.lock','a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return
            with open('/proc/1/fd/1','wb',buffering=0) as output, open('/proc/1/fd/2','wb',buffering=0) as errors:
                follow(args.path,output,errors,once=args.once)
    else:
        follow(args.path,sys.stdout.buffer,sys.stderr.buffer,once=args.once)


if __name__=='__main__':
    main()
