import win32process as process
import time
from ctypes import *
import ctypes.wintypes as wtypes
import win32gui as wgui
TH_PATH = "D:\\Games\\Steam\\steamapps\\common\\th16tr\\th16.exe"

startObj = process.STARTUPINFO()
print(startObj)
procHandle, threadHandle, pid, threadId = process.CreateProcess(TH_PATH, None, None, None, 8, 8, None, None, startObj)

time.sleep(14)
OpenProcess = windll.kernel32.OpenProcess
ReadProcessMemory = windll.kernel32.ReadProcessMemory
CloseHandle = windll.kernel32.CloseHandle


PROCESS_ALL_ACCESS = 0x1F0FFF

score_addr = c_void_p(0x0049E440)

buffer = wtypes.LPVOID(0)
print(buffer)
bufferSize = 4# sizeof(buffer)
print("Buffer size:", bufferSize, "bytes")
bytesRead = c_ulong(0)


processHandle = OpenProcess(PROCESS_ALL_ACCESS, False, pid)

if ReadProcessMemory(processHandle, score_addr, byref(buffer), bufferSize, byref(bytesRead)):
    print("Momory read success:", buffer)
else:
    print("Memory read failed")
    print(bytesRead.value)
    print(buffer.value)
print("err", windll.kernel32.GetLastError())
time.sleep(10)
