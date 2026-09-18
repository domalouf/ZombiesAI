"""The real game: adapters that turn spec actions into input the engine believes, and pixels back out.

Nothing here simulates anything. `dispatch.py` is the backend-agnostic half -- which transitions a factored
action implies, and when to send them -- and is fully testable; `uinput.py` is the Linux half that hands
those transitions to the kernel. Screen capture lives in `demos/x11_capture.py`, because recording a human
and observing the agent read the same pixels the same way.
"""
