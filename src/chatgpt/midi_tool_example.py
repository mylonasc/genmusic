import time
from midi_tool import MidiToolNB  # if you put the code into midi_tool_nb.py

tool = MidiToolNB()
handle = tool.open_digitakt()

print("IN :", handle.input_name)
print("OUT:", handle.output_name)

try:
    while True:
        # do other work here (UI/audio/logic)
        # msgs = tool.get_messages(handle)
        # for m in msgs:
        #     print("RX:", m)

        # send something occasionally
        # tool.send_cc(handle, control=1, value=5, channel=1)
        for c in [10,11,12,2]:
            for n in [65,66,67,70,75]:
                tool.send_note_on(handle, n, 100, channel=c)
                time.sleep(0.1)
                tool.send_note_off(handle,  n, 100, channel=c)
                print(c)
        time.sleep(0.01)  # your app tick
finally:
    handle.close()
