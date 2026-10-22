"""Process-safe model writer lock, using only the Python 3.10 standard library."""
from contextlib import contextmanager
import os
from pathlib import Path


class ModelBusyError(RuntimeError):
    pass


@contextmanager
def model_writer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory/".model-write.lock").open("a+b")
    locked = False
    try:
        if stream.seek(0, 2) == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise ModelBusyError("Một tiến trình khác đang huấn luyện hoặc cập nhật model.") from exc
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
