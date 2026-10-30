import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.batch_io import Tee,save_state


class BatchIOTests(unittest.TestCase):
    def test_transient_memory_error_retries_and_writes_real_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'software-fixture.json'
            original=Path.open
            calls=[]
            def open_file(item,*args,**kwargs):
                calls.append(item)
                if len(calls)<3:
                    raise OSError(12,'software fixture: transient error')
                return original(item,*args,**kwargs)
            with patch.object(Path,'open',open_file),patch('evaluation.batch_io.time.sleep'):
                save_state(path,dict(status='EXTRACTING'))
            self.assertEqual(json.loads(path.read_text())['status'],'EXTRACTING')
            self.assertEqual(len(calls),3)

    def test_persistent_error_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'software-fixture.json'
            path.write_text('{"status":"EXTRACTING"}')
            with patch.object(Path,'open',side_effect=OSError(12,'fixture')),patch('evaluation.batch_io.time.sleep'):
                with self.assertRaises(OSError):
                    save_state(path,dict(status='EXTRACTION_COMPLETED'),attempts=2)
            self.assertEqual(json.loads(path.read_text())['status'],'EXTRACTING')

    def test_tee_keeps_file_and_docker_console(self):
        primary,console=io.StringIO(),io.StringIO()
        tee=Tee(primary,console)
        tee.write('actual-message-placeholder\n')
        self.assertEqual(primary.getvalue(),console.getvalue())


if __name__=='__main__':
    unittest.main()
