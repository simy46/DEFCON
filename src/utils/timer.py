import time

class Timer:
    def __init__(self, msg=""):
        self.msg = msg
    
    def __enter__(self):
        print(f"\n[TIMER START] {self.msg}")
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time()
        print(f"[TIMER END] {self.msg} ({end - self.start:.2f} sec)\n")
        return False