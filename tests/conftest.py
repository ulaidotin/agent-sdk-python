import signal

import pytest
from fakeserver import fake  # noqa: F401  (the fixture)

TEST_TIMEOUT = 30


@pytest.fixture(autouse=True)
def _fail_hung_tests():
    """A hung test fails after TEST_TIMEOUT seconds instead of stalling the run."""

    def hung(signum, frame):
        raise TimeoutError(f"test did not finish within {TEST_TIMEOUT}s")

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(TEST_TIMEOUT)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)
