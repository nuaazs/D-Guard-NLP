import time
def timeit(func):
    def wrapper(*args,**kwargs):
        start=time.time()
        func(*args,**kwargs)
        print(f">>> Time used: {time.time()-start:.2f}s")
    return wrapper
