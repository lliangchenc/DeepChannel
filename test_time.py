import time
import gc
import os
import subprocess
import multiprocessing as mp
import pyrouge
import codecs
import shutil
import os
import re
from subprocess import check_output, CalledProcessError



def run_cmd_(a):
    out = subprocess.check_output(['pwd'])
    a.send(out)

def run_cmd():
    a, b = mp.Pipe()
    tic = time.time()
    p = mp.Process(target=run_cmd_, args=(a,))
    p.start()
    p.join()
    print(b.recv())
    print(time.time() - tic)

def test_mp():
    run_cmd()
    print('hello')
    run_cmd()

def test():
    #input()
    #cmds = ['ls']
    a = np.zeros((1,))
    tic = time.time()
    subprocess.call(['pwd'])
    #os.system('ls')
    print(time.time() - tic)

if __name__ == '__main__':
    mp.set_start_method('forkserver') # use fork server to take in charge of fork every time
    _ = [1] * 511111111
    #test_mp()
    from pyrouge import Rouge155
    import numpy as np
    test()
    r = Rouge155()
    r.test()
    r.test()
    r.score('abc', {'A':'abc'})
    r.score('abc', {'A':'abc'})

