import math

def entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return - p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


def calc_gain(lables, indices):
    max_index = max(indices)

    index_map = [[0, 0] for x in range(max_index + 1)]
    i = 0
    for index in indices:
        if lables[i] == 1:
            index_map[index][0] += 1
        else:
            index_map[index][1] += 1
        i += 1
    index_map = filter(lambda t: t[0] > 0 or t[1] > 0, index_map)

    # print "index map:", index_map

    total_positive_count = 0.0
    total_count = 0.0
    for x in index_map:
        total_positive_count += x[0]
        total_count += (x[0] + x[1])

    total_entropy = entropy(total_positive_count / total_count)

    conditional_entropy = 0.0
    for x in index_map:
        index_count = float(x[0] + x[1])
        conditional_entropy += (index_count / total_count) * entropy(x[0] / index_count)

    return total_entropy - conditional_entropy