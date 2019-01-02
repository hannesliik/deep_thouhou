from ctypes import *


def write_memory(handle, address, value, bytes):
    """Writes :bytes of :value to the given :address of the process :handle"""
    written_memory = POINTER(c_uint32)
    num = c_uint32(value)
    addr = addressof(num)
    ptr = cast(addr, written_memory)
    print(written_memory)
    bytes_written = c_ulong(0)
    if windll.kernel32.WriteProcessMemory(handle, address, ptr, bytes, byref(bytes_written)):
        print("Wrote memory")
        return bytes_written
    else:
        print("Failed to write to memory. Error:", windll.kernel32.GetLastError())
        return bytes_written


def read_memory(handle, address, buffer, bytes_to_read):
    """"Reads the given memory address and stores it in the given buffer.
     The buffer must be of the appropriate type"""
    bytes_read = c_ulong(0)
    if windll.kernel32.ReadProcessMemory(handle, address, byref(buffer), bytes_to_read, byref(bytes_read)):
        return buffer, bytes_read
    else:
        print("Memory read failed, Error:", windll.kernel32.GetLastError())
        exit()
        return buffer, bytes_read
