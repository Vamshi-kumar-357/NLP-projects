from collections import Counter

def extract_ngrams(tokens, min_n, max_n, count_mode=False):
    """Extract ngrams as tuples from a list of tokens."""
    ngrams = Counter() if count_mode else set()
    for l in range(len(tokens)):
        for d in range(min_n, max_n + 1):
            if l + d <= len(tokens):
                ngram = tuple(tokens[l:l + d])
                if count_mode:
                    ngrams[ngram] += 1
                else:
                    ngrams.add(ngram)
    return ngrams


# =========================
# 1️⃣ Presence Kernel
# =========================
def computeKernelMatrix_presence(start_gram, end_gram, samples):
    """Kernel based on presence/absence of n-grams (train-train)."""
    n = len(samples)
    K = [[0] * n for _ in range(n)]

    ngram_sets = [extract_ngrams(s, start_gram, end_gram, count_mode=False) for s in samples]

    for i in range(n):
        for j in range(i, n):
            common = ngram_sets[i] & ngram_sets[j]
            K[i][j] = K[j][i] = len(common)
        if (i + 1) % 500 == 0:
            print(f"Computed kernel up to row {i}")
    return K


def computeKernelMatrix_presence_test(start_gram, end_gram, train_samples, test_samples):
    """Presence kernel (test vs train)."""
    train_sets = [extract_ngrams(s, start_gram, end_gram, count_mode=False) for s in train_samples]
    test_sets = [extract_ngrams(s, start_gram, end_gram, count_mode=False) for s in test_samples]

    K = [[0] * len(train_samples) for _ in range(len(test_samples))]

    for i, tset in enumerate(test_sets):
        for j, trset in enumerate(train_sets):
            K[i][j] = len(tset & trset)
        if (i + 1) % 500 == 0:
            print(f"Computed kernel up to row {i}")
    return K


# =========================
# 2️⃣ Intersection Kernel
# =========================
def computeKernelMatrix_intersection(start_gram, end_gram, samples):
    """Intersection kernel counts min overlaps (train-train)."""
    n = len(samples)
    K = [[0] * n for _ in range(n)]

    ngram_counts = [extract_ngrams(s, start_gram, end_gram, count_mode=True) for s in samples]

    for i in range(n):
        for j in range(i, n):
            # intersection: sum of min counts for shared ngrams
            common_keys = set(ngram_counts[i]) & set(ngram_counts[j])
            K[i][j] = K[j][i] = sum(min(ngram_counts[i][k], ngram_counts[j][k]) for k in common_keys)
        if (i + 1) % 500 == 0:
            print(f"Computed kernel up to row {i}")
    return K


def computeKernelMatrix_intersection_test(start_gram, end_gram, train_samples, test_samples):
    """Intersection kernel (test vs train)."""
    train_counts = [extract_ngrams(s, start_gram, end_gram, count_mode=True) for s in train_samples]
    test_counts = [extract_ngrams(s, start_gram, end_gram, count_mode=True) for s in test_samples]

    K = [[0] * len(train_samples) for _ in range(len(test_samples))]

    for i, tcount in enumerate(test_counts):
        for j, trcount in enumerate(train_counts):
            common_keys = set(tcount) & set(trcount)
            K[i][j] = sum(min(tcount[k], trcount[k]) for k in common_keys)
        if (i + 1) % 500 == 0:
            print(f"Computed kernel up to row {i}")
    return K


# =========================
# 3️⃣ Spectrum Kernel
# =========================
def computeKernelMatrix_spectrum(start_gram, end_gram, samples):
    """Spectrum kernel sums count products (train-train)."""
    n = len(samples)
    K = [[0] * n for _ in range(n)]

    ngram_counts = [extract_ngrams(s, start_gram, end_gram, count_mode=True) for s in samples]

    for i in range(n):
        for j in range(i, n):
            common_keys = set(ngram_counts[i]) & set(ngram_counts[j])
            K[i][j] = K[j][i] = sum(ngram_counts[i][k] * ngram_counts[j][k] for k in common_keys)
        if (i + 1) % 500 == 0:
            print(f"Computed kernel up to row {i}")
    return K


def computeKernelMatrix_spectrum_test(start_gram, end_gram, train_samples, test_samples):
    """Spectrum kernel (test vs train)."""
    train_counts = [extract_ngrams(s, start_gram, end_gram, count_mode=True) for s in train_samples]
    test_counts = [extract_ngrams(s, start_gram, end_gram, count_mode=True) for s in test_samples]

    K = [[0] * len(train_samples) for _ in range(len(test_samples))]

    for i, tcount in enumerate(test_counts):
        for j, trcount in enumerate(train_counts):
            common_keys = set(tcount) & set(trcount)
            K[i][j] = sum(tcount[k] * trcount[k] for k in common_keys)
        if (i + 1) % 500 == 0:
            print(f"Computed kernel up to row {i}")
    return K
