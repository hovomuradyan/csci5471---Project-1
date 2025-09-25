import math
import csv
import heapq

def load_bigrams(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    header = rows[0][1:]
    header[0] = ' '
    table = {}
    for row in rows[1:]:
        if not row:
            continue
        first = row[0] if row[0] else ' '
        counts = [float(x) if x else 0.0 for x in row[1:]]
        total = sum(counts)
        for ch, c in zip(header, counts):
            prob = (c / total) if total > 0 else 1e-12
            table[(first, ch)] = math.log(max(prob, 1e-12))
    return table

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def score_bigram(prev, curr, table):
    return table.get((prev, curr), math.log(1e-12))

def load_dictionary(path='/usr/share/dict/words'):
    with open(path) as f:
        words = set(line.strip().upper() for line in f if line.strip().isalpha())
    return words

def beam_search(x, table, dictionary, beam_width=20):
    charset = [32] + list(range(65, 91)) + list(range(97, 123))
    charset = [chr(c) if c != 32 else ' ' for c in charset]
    beam = [('', '', 0.0)]
    for i in range(len(x)):
        new_beam = []
        for p1, p2, score in beam:
            prev1 = p1[-1] if p1 else ' '
            prev2 = p2[-1] if p2 else ' '
            for c1 in charset:
                c2 = chr(x[i] ^ ord(c1))
                if c2 not in charset:
                    continue
                new_p1 = p1 + c1
                new_p2 = p2 + c2
                s1 = score_bigram(prev1, c1, table)
                s2 = score_bigram(prev2, c2, table)
                penalty = 0.0
                if len(new_p1) % 5 == 0:
                    last_word1 = new_p1[-5:].strip().upper()
                    last_word2 = new_p2[-5:].strip().upper()
                    if last_word1 not in dictionary:
                        penalty += 5.0
                    if last_word2 not in dictionary:
                        penalty += 5.0
                total_score = score + s1 + s2 - penalty
                new_beam.append((new_p1, new_p2, total_score))
        if not new_beam:
            return beam[0][0], beam[0][1]
        beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[2])
    return beam[0][0], beam[0][1]

def main():
    with open('source1.txt','rb') as f:
        ct = f.read()
    c1, c2 = ct[:1024], ct[1024:]
    x = xor_bytes(c1, c2)
    table = load_bigrams('ftable2.csv')
    dictionary = load_dictionary()
    p1, p2 = beam_search(x, table, dictionary, beam_width=30)
    with open('plain1.txt','w') as f:
        f.write(p1)
    with open('plain2.txt','w') as f:
        f.write(p2)

if __name__ == "__main__":
    main()

