from RandomCipher import Decoder
from time import time

encoded = open('encoded_text.txt').read()
answer = open('temp.txt').read()

count = 1
total = 0
acc = 0
for i in range(count):
    # print(i)
    d = Decoder(encoded)
    start = time()
    r=d.decode()
    if r == answer:
        acc+=1
    end = time()
    total += end-start

print('avg time: %d\nacc: %d'%(total/count,acc/count))